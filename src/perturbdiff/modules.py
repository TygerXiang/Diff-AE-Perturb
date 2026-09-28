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

    def forward(
        self,
        z_sem: torch.Tensor,
        perturbation_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the per-perturbation latent offset and the basal latent.

        This method used to also return ``z_pert = z_sem``, which made it look as though the
        perturbation offset conditioned the denoiser when in fact it did not.  Building the
        conditioning vector is the job of ``PerturbDiffusionModel.encode_condition``, which is
        now the single place that defines ``z_pert``.
        """
        delta_base = self.embedding(perturbation_ids)
        delta = delta_base
        # z_basal is the latent with the perturbation offset removed; it is used by the
        # optional semantic-alignment regulariser and for interpretation.
        z_basal = z_sem - delta
        return {
            "z_basal": z_basal,
            "z_sem": z_sem,
            "delta_base": delta_base,
            "delta": delta,
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
        # At inference we explicitly "remove source, add target" in latent
        # space. This is the point where perturbation arithmetic becomes a real
        # intervention instead of just an interpretation.
        z_basal = z_sem - source_delta
        if guide_scales is not None:
            if guide_scales.ndim == 1:
                target_delta = target_delta * guide_scales.unsqueeze(-1)
            else:
                raise ValueError("guide_scales must be a 1D tensor")
        return z_basal + target_delta


class GeneralizedSigmoid(nn.Module):
    """Target-specific scalar nonlinearity for activity conditioning."""

    def __init__(self, n_targets: int, non_linearity: str = "sigm") -> None:
        super().__init__()
        self.non_linearity = non_linearity
        self.beta = nn.Parameter(torch.ones(1, n_targets + 1))
        self.bias = nn.Parameter(torch.zeros(1, n_targets + 1))

    def forward(self, x: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        target_ids = target_ids.long()
        if self.non_linearity == "linear":
            return x
        bias = self.bias[0][target_ids]
        beta = self.beta[0][target_ids]
        c0 = bias.sigmoid()
        if self.non_linearity == "logsigm":
            scaled = (torch.log1p(x.clamp_min(0.0)) * beta + bias).sigmoid() - c0
        elif self.non_linearity == "sigm":
            scaled = (x * beta + bias).sigmoid() - c0
        else:
            raise ValueError(f"Unsupported doser_type: {self.non_linearity!r}")
        return scaled.clamp_min(0.0)


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
    """Vector denoiser epsilon_theta(x_t, t, z_pert, activity_embedding)."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...],
        time_embed_dim: int | None = None,
        activity_condition_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim or latent_dim
        self.activity_condition_dim = activity_condition_dim
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.network = make_mlp(
            n_genes + latent_dim + self.time_embed_dim + activity_condition_dim,
            hidden_dims,
            n_genes,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        z_pert: torch.Tensor,
        activity_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        inputs = [x_t, t_emb, z_pert]
        if self.activity_condition_dim > 0:
            if activity_embedding is None:
                activity_embedding = x_t.new_zeros(x_t.shape[0], self.activity_condition_dim)
            if activity_embedding.shape != (x_t.shape[0], self.activity_condition_dim):
                raise ValueError(
                    "activity_embedding must have shape "
                    f"{(x_t.shape[0], self.activity_condition_dim)}"
                )
            inputs.append(activity_embedding)
        return self.network(torch.cat(inputs, dim=-1))
