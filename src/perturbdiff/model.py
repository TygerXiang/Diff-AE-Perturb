from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .diffusion import GaussianDiffusion
from .losses import gears_autofocus_direction_loss, weighted_mse_loss
from .modules import ConditionalDenoiserMLP, PerturbationModule, SemanticEncoder


class PerturbDiffusionModel(nn.Module):
    """Conditional DDIM model with scGen-style latent perturbation deltas."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SemanticEncoder(
            n_genes=config.n_genes,
            latent_dim=config.latent_dim,
            hidden_dims=config.encoder_hidden_dims,
            dropout=config.dropout,
        )
        self.perturbation = PerturbationModule(
            n_perturbations=config.n_perturbations,
            latent_dim=config.latent_dim,
        )
        self.denoiser = ConditionalDenoiserMLP(
            n_genes=config.n_genes,
            latent_dim=config.latent_dim,
            hidden_dims=config.denoiser_hidden_dims,
            dropout=config.dropout,
        )
        self.diffusion = GaussianDiffusion(config.diffusion)
        self.register_buffer(
            "signature_gene_weights",
            torch.ones(config.n_perturbations + 1, config.n_genes),
            persistent=False,
        )
        self.register_buffer(
            "control_expression_mean",
            torch.zeros(config.n_genes),
            persistent=False,
        )
        self._has_control_expression_mean = False

        if config.losses.reconstruction_loss not in {"mse", "gears"}:
            raise ValueError(
                "Unsupported reconstruction_loss: "
                f"{config.losses.reconstruction_loss!r}"
            )

    def set_signature_gene_weights(self, signature_gene_weights: torch.Tensor | None) -> None:
        with torch.no_grad():
            if signature_gene_weights is None:
                self.signature_gene_weights.fill_(1.0)
                return
            if signature_gene_weights.shape != self.signature_gene_weights.shape:
                raise ValueError(
                    "signature_gene_weights must have shape "
                    f"{tuple(self.signature_gene_weights.shape)}"
                )
            self.signature_gene_weights.copy_(
                signature_gene_weights.to(
                    device=self.signature_gene_weights.device,
                    dtype=self.signature_gene_weights.dtype,
                )
            )

    def set_control_expression_mean(self, control_expression_mean: torch.Tensor | None) -> None:
        with torch.no_grad():
            if control_expression_mean is None:
                self.control_expression_mean.zero_()
                self._has_control_expression_mean = False
                return
            if control_expression_mean.shape != self.control_expression_mean.shape:
                raise ValueError(
                    "control_expression_mean must have shape "
                    f"{tuple(self.control_expression_mean.shape)}"
                )
            self.control_expression_mean.copy_(
                control_expression_mean.to(
                    device=self.control_expression_mean.device,
                    dtype=self.control_expression_mean.dtype,
                )
            )
            self._has_control_expression_mean = True

    @torch.no_grad()
    def initialize_perturbation_from_latent_shift(
        self,
        z_sem: torch.Tensor,
        perturbation_ids: torch.Tensor,
    ) -> dict[str, dict[int, int]]:
        if z_sem.ndim != 2:
            raise ValueError("z_sem must be a 2D tensor")
        if z_sem.shape[0] != perturbation_ids.shape[0]:
            raise ValueError("z_sem and perturbation_ids must have matching batch size")

        control_mask = perturbation_ids == 0
        if not control_mask.any():
            raise ValueError("control cells are required to initialize perturbation embeddings")
        control_mean = z_sem[control_mask].mean(dim=0)

        target_means = control_mean.repeat(self.config.n_perturbations + 1, 1)
        counts: dict[int, int] = {0: int(control_mask.sum().item())}
        for target_id in range(1, self.config.n_perturbations + 1):
            mask = perturbation_ids == target_id
            count = int(mask.sum().item())
            counts[target_id] = count
            if count == 0:
                continue
            target_means[target_id] = z_sem[mask].mean(dim=0)

        self.perturbation.initialize_from_latent_shift(control_mean, target_means)
        return {"counts": counts}

    def encode_condition(
        self,
        x0: torch.Tensor,
        perturbation_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_sem = self.encoder(x0)
        pert = self.perturbation(z_sem, perturbation_ids)
        return {**pert}

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        z_pert: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(x_t, timesteps, z_pert)

    def training_losses(
        self,
        x0: torch.Tensor,
        perturbation_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if timesteps is None:
            timesteps = self.diffusion.sample_timesteps(x0.shape[0], x0.device)
        condition = self.encode_condition(x0, perturbation_ids)
        x_t, noise = self.diffusion.q_sample(x0, timesteps, noise=noise)
        prediction = self.denoiser(x_t, timesteps, condition["z_pert"])
        target = x0 if self.diffusion.prediction_type == "x0" else noise
        gene_weights = self.signature_gene_weights.index_select(0, perturbation_ids)
        eval_mse_loss = weighted_mse_loss(prediction, target, gene_weights)
        if self.config.losses.reconstruction_loss == "mse":
            diffusion_loss = eval_mse_loss
        else:
            prediction_x0, _ = self.diffusion.prediction_to_x0_and_eps(
                x_t,
                timesteps,
                prediction,
            )
            control_mean = (
                self.control_expression_mean
                if self._has_control_expression_mean
                else None
            )
            diffusion_loss = gears_autofocus_direction_loss(
                prediction_x0,
                x0,
                gene_weights,
                control_mean=control_mean,
                gamma=self.config.losses.autofocus_gamma,
                direction_lambda=self.config.losses.direction_lambda,
            )
        semantic_align_loss = prediction.new_zeros(())
        direction_align_loss = prediction.new_zeros(())
        if self.config.losses.lambda_semantic_align > 0:
            control_mask = perturbation_ids == 0
            pert_mask = perturbation_ids != 0
            if control_mask.any() and pert_mask.any():
                z_control_mean = condition["z_basal"][control_mask].mean(dim=0, keepdim=True)
                semantic_align_loss = (condition["z_basal"][pert_mask] - z_control_mean).square().mean()
        if self.config.losses.lambda_direction_align > 0:
            if target_ids is None:
                target_ids = perturbation_ids
            valid = target_ids != 0
            if valid.sum().item() >= 2:
                delta = condition["delta"][valid]
                t = target_ids[valid]
                cos = torch.nn.functional.cosine_similarity(
                    delta.unsqueeze(1),
                    delta.unsqueeze(0),
                    dim=-1,
                )
                same_target = t.unsqueeze(1).eq(t.unsqueeze(0))
                upper = torch.triu(torch.ones_like(same_target, dtype=torch.bool), diagonal=1)
                pair_mask = same_target & upper
                if pair_mask.any():
                    direction_align_loss = (1.0 - cos[pair_mask]).mean()
        total_loss = (
            diffusion_loss
            + self.config.losses.lambda_semantic_align * semantic_align_loss
            + self.config.losses.lambda_direction_align * direction_align_loss
        )
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "eval_mse_loss": eval_mse_loss,
            "semantic_align_loss": semantic_align_loss,
            "direction_align_loss": direction_align_loss,
            "z_basal": condition["z_basal"],
            "z_sem": condition["z_sem"],
            "z_pert": condition["z_pert"],
            "delta": condition["delta"],
            "prediction": prediction,
        }

    @torch.no_grad()
    def sample_counterfactual(
        self,
        x_ctrl: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        guide_scales: torch.Tensor | None = None,
        steps: int = 100,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_sem = self.encoder(x_ctrl)
        z_cf = self.perturbation.counterfactual_latent(
            z_sem,
            target_ids,
            source_ids=torch.zeros_like(target_ids),
            guide_scales=guide_scales,
        )
        return self.diffusion.ddim_sample(
            self.denoiser,
            shape=x_ctrl.shape,
            z_cond=z_cf,
            steps=steps,
            noise=noise,
        )
