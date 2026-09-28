from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .diffusion import GaussianDiffusion
from .guide_features import GuideStrengthEncoder
from .losses import gears_autofocus_direction_loss, weighted_mse_loss
from .modules import ConditionalDenoiserMLP, GeneralizedSigmoid, PerturbationModule, SemanticEncoder


class PerturbDiffusionModel(nn.Module):
    """Conditional DDIM model with learnable latent perturbation deltas."""

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
        if config.guide_encoder.enabled:
            if config.guide_encoder.feature_dim < 1:
                raise ValueError("guide_encoder.feature_dim must be >= 1 when enabled")
            if config.guide_encoder.activity_source not in {"predicted", "direct"}:
                raise ValueError("guide_encoder.activity_source must be 'predicted' or 'direct'")
            self.guide_strength = (
                GuideStrengthEncoder(
                    feature_dim=config.guide_encoder.feature_dim,
                    architecture=config.guide_encoder.architecture,
                    hidden_dims=config.guide_encoder.hidden_dims,
                    dropout=config.guide_encoder.dropout,
                    sequence_length=config.guide_encoder.sequence_length,
                    cnn_channels=config.guide_encoder.cnn_channels,
                    cnn_kernel_size=config.guide_encoder.cnn_kernel_size,
                    cnn_dropout=config.guide_encoder.cnn_dropout,
                )
                if config.guide_encoder.activity_source == "predicted"
                else None
            )
            self.guide_doser = GeneralizedSigmoid(
                n_targets=config.n_perturbations,
                non_linearity=config.guide_encoder.doser_type,
            )
        else:
            self.guide_strength = None
            self.guide_doser = None
        self.activity_condition_dim = (
            int(getattr(config.guide_encoder, "activity_condition_dim", 0))
            if config.guide_encoder.enabled
            and bool(getattr(config.guide_encoder, "explicit_condition", False))
            else 0
        )
        if self.activity_condition_dim < 0:
            raise ValueError("guide_encoder.activity_condition_dim must be >= 0")
        if self.activity_condition_dim > 0:
            self.activity_condition = nn.Sequential(
                nn.Linear(1, self.activity_condition_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.activity_condition_dim, self.activity_condition_dim, bias=False),
            )
        else:
            self.activity_condition = None
        self.denoiser = ConditionalDenoiserMLP(
            n_genes=config.n_genes,
            latent_dim=config.latent_dim,
            hidden_dims=config.denoiser_hidden_dims,
            activity_condition_dim=self.activity_condition_dim,
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

    def encode_condition(
        self,
        x0: torch.Tensor,
        perturbation_ids: torch.Tensor,
        *,
        target_ids: torch.Tensor | None = None,
        guide_features: torch.Tensor | None = None,
        guide_activity_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build every conditioning tensor the denoiser may consume.

        The counterfactual re-parameterisation

            z_pert = z_control + delta[perturbation]        (control cells keep their own z_sem)

        is applied **unconditionally**.  It used to sit behind an early ``return`` that fired
        whenever ``guide_doser is None``, which silently reduced the conditioning to
        ``z_pert = z_sem = encoder(x0)`` for every target-only run: the denoiser then saw an
        encoding of the very cell it had to denoise, so the per-perturbation offset never
        reached it and was never trained (see reports/diffae_architecture_diagnosis.md).
        Only the guide/activity branch is optional now.
        """
        z_sem = self.encoder(x0)
        # The perturbation module returns several latent views because different
        # losses care about different interpretations of the same cell state.
        pert = self.perturbation(z_sem, perturbation_ids)
        guide_strength_loss = z_sem.new_zeros(())
        strength = None
        activity_value = None
        activity_embedding = None

        if self.guide_doser is not None:
            if guide_features is None:
                strength = torch.ones(x0.shape[0], device=x0.device, dtype=x0.dtype)
            else:
                guide_features = guide_features.to(device=x0.device, dtype=x0.dtype)
                if self.config.guide_encoder.activity_source == "direct":
                    if guide_features.ndim == 1:
                        strength = guide_features
                    elif guide_features.ndim == 2 and guide_features.shape[1] >= 1:
                        strength = guide_features[:, 0]
                    else:
                        raise ValueError("direct activity mode expects guide_features with shape (batch,) or (batch, >=1)")
                    strength = strength.clamp_min(0.0)
                else:
                    if self.guide_strength is None:
                        raise RuntimeError("guide_strength encoder is required for predicted activity mode")
                    strength = self.guide_strength(guide_features)
            if self.config.guide_encoder.lambda_strength > 0 and guide_activity_targets is not None:
                target_strength = guide_activity_targets.to(device=x0.device, dtype=x0.dtype).clamp_min(0.0)
                guide_strength_loss = torch.nn.functional.mse_loss(strength, target_strength)

            activity_ids = target_ids if target_ids is not None else perturbation_ids
            activity_value = self.guide_doser(strength, activity_ids)
            activity_embedding = self._build_activity_embedding(activity_value, perturbation_ids)

        if activity_value is not None and bool(
            getattr(self.config.guide_encoder, "activity_scales_delta", False)
        ):
            pert["delta"] = pert["delta_base"] * activity_value.unsqueeze(-1)
        else:
            pert["delta"] = pert["delta_base"]

        pert["z_basal"] = z_sem - pert["delta"]
        z_control = self._build_control_anchor(z_sem, pert["delta"], perturbation_ids)
        pert["z_control"] = z_control
        if str(getattr(self.config, "control_reference", "per_cell")) == "encoder_self":
            # Legacy ablation only.  This conditions the denoiser on the encoding of the very
            # cell it has to denoise, so the perturbation offset never reaches it and is never
            # trained; it is what produced the delta-correlation ~0.03 checkpoint.  Kept as an
            # explicit, named mode so the failure stays reproducible and testable.
            pert["z_pert"] = z_sem
        else:
            # Control cells condition on their own latent; every perturbed cell conditions on a
            # control-anchored latent shifted by its perturbation offset.
            pert["z_pert"] = torch.where(
                perturbation_ids.unsqueeze(-1) == 0,
                z_sem,
                z_control + pert["delta"],
            )
        pert["strength"] = strength
        pert["scaled_strength"] = activity_value
        pert["activity_embedding"] = activity_embedding
        pert["guide_strength_loss"] = guide_strength_loss
        return {**pert}

    def _build_activity_embedding(
        self,
        activity_value: torch.Tensor,
        perturbation_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.activity_condition is None:
            return None
        if activity_value.ndim != 1:
            raise ValueError("activity_value must be a 1D tensor")
        active = (perturbation_ids != 0).to(dtype=activity_value.dtype, device=activity_value.device)
        activity_value = activity_value * active
        return self.activity_condition(activity_value.unsqueeze(-1))

    def _build_control_anchor(
        self,
        z_sem: torch.Tensor,
        delta: torch.Tensor,
        perturbation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Anchor every perturbed cell to a control state in latent space.

        ``config.control_reference`` decides how, and it must match the inference path
        (``sample_counterfactual``) or the denoiser is asked to extrapolate:

        ``per_cell``    each row is paired with one control cell's latent (deterministic
                        round-robin, so runs are reproducible).  This is the default and it
                        matches what inference does with a batch of control cells.
        ``batch_mean``  every row shares the batch control mean, which is what the original
                        implementation did.  Selecting it also switches inference to a single
                        mean control latent so the two sides stay consistent.
        """
        reference = str(getattr(self.config, "control_reference", "per_cell"))
        control_mask = perturbation_ids == 0
        if reference in {"per_cell", "encoder_self"} and bool(control_mask.any()):
            control_z = z_sem[control_mask]
            indices = torch.arange(z_sem.shape[0], device=z_sem.device) % control_z.shape[0]
            return control_z.index_select(0, indices)
        if bool(control_mask.any()):
            control_mean = z_sem[control_mask].mean(dim=0, keepdim=True)
        else:
            # No control cell in this batch: fall back to the offset-removed mean, which is
            # the best available estimate of the basal state.
            control_mean = (z_sem - delta).mean(dim=0, keepdim=True)
        return control_mean.expand_as(z_sem)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        z_pert: torch.Tensor,
        activity_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.denoiser(x_t, timesteps, z_pert, activity_embedding)

    def training_losses(
        self,
        x0: torch.Tensor,
        perturbation_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        guide_features: torch.Tensor | None = None,
        guide_activity_targets: torch.Tensor | None = None,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if timesteps is None:
            timesteps = self.diffusion.sample_timesteps(x0.shape[0], x0.device)
        condition = self.encode_condition(
            x0,
            perturbation_ids,
            target_ids=target_ids,
            guide_features=guide_features,
            guide_activity_targets=guide_activity_targets,
        )
        x_t, noise = self.diffusion.q_sample(x0, timesteps, noise=noise)
        prediction = self.denoiser(
            x_t,
            timesteps,
            condition["z_pert"],
            condition["activity_embedding"],
        )
        # Diffusion training can target either the original clean sample x0 or
        # the sampled noise eps. The config decides which learning problem we
        # pose to the denoiser.
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
                # Encourage "remove perturbation" latents to sit near the
                # control manifold, which makes delta easier to interpret as a
                # perturbation-specific shift.
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
                # Same-target guides do not need identical magnitudes, but they
                # often should point in similar directions in latent space.
                same_target = t.unsqueeze(1).eq(t.unsqueeze(0))
                upper = torch.triu(torch.ones_like(same_target, dtype=torch.bool), diagonal=1)
                pair_mask = same_target & upper
                if pair_mask.any():
                    direction_align_loss = (1.0 - cos[pair_mask]).mean()
        total_loss = (
            diffusion_loss
            + self.config.losses.lambda_semantic_align * semantic_align_loss
            + self.config.losses.lambda_direction_align * direction_align_loss
            + self.config.guide_encoder.lambda_strength * condition["guide_strength_loss"]
        )
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "eval_mse_loss": eval_mse_loss,
            "semantic_align_loss": semantic_align_loss,
            "direction_align_loss": direction_align_loss,
            "guide_strength_loss": condition["guide_strength_loss"],
            "z_basal": condition["z_basal"],
            "z_sem": condition["z_sem"],
            "z_pert": condition["z_pert"],
            "delta": condition["delta"],
            "activity_embedding": condition["activity_embedding"],
            "prediction": prediction,
        }

    @torch.no_grad()
    def initialize_guide_strength_directions_from_perturbation_embeddings(
        self,
        target_ids: torch.Tensor,
    ) -> None:
        if self.guide_doser is None:
            return
        ids = target_ids.to(
            device=self.perturbation.embedding.weight.device,
            dtype=torch.long,
        ).clamp(min=0, max=self.config.n_perturbations)
        norms = self.perturbation.embedding.weight.detach().norm(dim=1)
        scale = norms.index_select(0, ids)
        ref = float(scale[ids != 0].mean().item()) if torch.any(ids != 0) else 1.0
        ref = max(ref, 1e-6)
        with torch.no_grad():
            self.guide_doser.beta.fill_(1.0 / ref)
            self.guide_doser.bias.zero_()

    @torch.no_grad()
    def sample_counterfactual(
        self,
        x_ctrl: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        guide_features: torch.Tensor | None = None,
        guide_scales: torch.Tensor | None = None,
        steps: int = 100,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_sem = self.encoder(x_ctrl)
        activity_value = None
        if self.guide_doser is not None:
            if guide_scales is not None:
                raise ValueError("guide_scales cannot be combined with guide_features activity conditioning")
            if guide_features is None:
                strength = torch.ones(x_ctrl.shape[0], device=x_ctrl.device, dtype=x_ctrl.dtype)
            else:
                guide_features = guide_features.to(device=x_ctrl.device, dtype=x_ctrl.dtype)
                if self.config.guide_encoder.activity_source == "direct":
                    if guide_features.ndim == 1:
                        strength = guide_features
                    elif guide_features.ndim == 2 and guide_features.shape[1] >= 1:
                        strength = guide_features[:, 0]
                    else:
                        raise ValueError("direct activity mode expects guide_features with shape (batch,) or (batch, >=1)")
                    strength = strength.clamp_min(0.0)
                else:
                    if self.guide_strength is None:
                        raise RuntimeError("guide_strength encoder is required for predicted activity mode")
                    strength = self.guide_strength(guide_features)
            activity_value = self.guide_doser(strength, target_ids)
        activity_embedding = (
            self._build_activity_embedding(activity_value, target_ids)
            if activity_value is not None
            else None
        )
        # Unlike training, inference starts from chosen control cells and explicitly builds the
        # target condition latent.  The control anchor must use the same convention as training
        # (config.control_reference), otherwise the denoiser is fed an out-of-distribution
        # conditioning vector.  "encoder_self" is a training-only ablation, so inference treats
        # it like "per_cell" -- which is exactly what the legacy checkpoints were evaluated with.
        if str(getattr(self.config, "control_reference", "per_cell")) == "batch_mean":
            z_source = z_sem.mean(dim=0, keepdim=True).expand_as(z_sem)
        else:
            z_source = z_sem
        z_cf = self.perturbation.counterfactual_latent(
            z_source,
            target_ids,
            source_ids=torch.zeros_like(target_ids),
            guide_scales=(
                activity_value
                if self.guide_doser is not None
                and bool(getattr(self.config.guide_encoder, "activity_scales_delta", False))
                else guide_scales
            ),
        )
        return self.diffusion.ddim_sample(
            self.denoiser,
            shape=x_ctrl.shape,
            z_cond=z_cf,
            activity_embedding=activity_embedding,
            steps=steps,
            noise=noise,
        )
