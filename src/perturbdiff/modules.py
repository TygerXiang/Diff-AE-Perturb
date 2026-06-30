from __future__ import annotations

import math

import torch
from torch import nn


def make_mlp(
    in_dim: int,
    hidden_dims: tuple[int, ...],
    out_dim: int,
    *,
    batch_norm: bool = False,
    dropout: float = 0.0,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, out_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class SemanticEncoder(nn.Module):
    """MLP encoder E_phi(x_0) -> z_sem."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = make_mlp(
            n_genes,
            hidden_dims,
            latent_dim,
            batch_norm=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PerturbationModule(nn.Module):
    """Target perturbation embeddings with control row fixed to zero."""

    def __init__(
        self,
        n_perturbations: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.n_perturbations = n_perturbations
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(
            n_perturbations + 1,
            latent_dim,
            padding_idx=0,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    @torch.no_grad()
    def set_base_embeddings(self, base_embeddings: torch.Tensor) -> None:
        if base_embeddings.shape != self.embedding.weight.shape:
            raise ValueError(
                "base_embeddings must have shape "
                f"{tuple(self.embedding.weight.shape)}"
            )
        self.embedding.weight.copy_(
            base_embeddings.to(
                device=self.embedding.weight.device,
                dtype=self.embedding.weight.dtype,
            )
        )
        self.embedding.weight[0].zero_()

    @torch.no_grad()
    def initialize_from_latent_shift(
        self,
        control_mean: torch.Tensor,
        target_means: torch.Tensor,
    ) -> None:
        if control_mean.ndim != 1:
            raise ValueError("control_mean must be a 1D tensor")
        if target_means.shape != self.embedding.weight.shape:
            raise ValueError(
                "target_means must have shape "
                f"{tuple(self.embedding.weight.shape)}"
            )
        delta = target_means.clone()
        delta = delta.to(device=self.embedding.weight.device, dtype=self.embedding.weight.dtype)
        control_mean = control_mean.to(device=self.embedding.weight.device, dtype=self.embedding.weight.dtype)
        delta = delta - control_mean.unsqueeze(0)
        delta[0].zero_()
        self.set_base_embeddings(delta)

    def forward(
        self,
        z_sem: torch.Tensor,
        perturbation_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta_base = self.embedding(perturbation_ids)
        delta = delta_base
        z_basal = z_sem - delta
        z_pert = z_sem
        return {
            "z_basal": z_basal,
            "z_sem": z_sem,
            "delta_base": delta_base,
            "delta": delta,
            "z_pert": z_pert,
        }

    def counterfactual_latent(
        self,
        z_sem: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        source_ids: torch.Tensor | None = None,
        guide_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_ids is None:
            source_ids = torch.zeros_like(target_ids)
        source_delta = self.embedding(source_ids)
        target_delta = self.embedding(target_ids)
        z_basal = z_sem - source_delta
        if guide_scales is not None:
            if guide_scales.ndim == 1:
                target_delta = target_delta * guide_scales.unsqueeze(-1)
            else:
                raise ValueError("guide_scales must be a 1D tensor")
        return z_basal + target_delta


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for integer diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        )
        exponent = exponent / max(half_dim - 1, 1)
        args = timesteps.float().unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ConditionalDenoiserMLP(nn.Module):
    """Vector denoiser epsilon_theta(x_t, t, z_pert)."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...],
        time_embed_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim or latent_dim
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.network = make_mlp(
            n_genes + latent_dim + self.time_embed_dim,
            hidden_dims,
            n_genes,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        z_pert: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        return self.network(torch.cat([x_t, t_emb, z_pert], dim=-1))
