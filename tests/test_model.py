from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff import DiffusionConfig, GuideFeatureEncoderConfig, ModelConfig, PerturbDiffusionModel, TrainingLossConfig
from perturbdiff.data import build_signature_gene_weights
from perturbdiff.guide_features import guide_cnn_feature_dim
from perturbdiff.losses import gears_autofocus_direction_loss, weighted_mse_loss


def small_config() -> ModelConfig:
    return ModelConfig(
        n_genes=20,
        n_perturbations=3,
        latent_dim=8,
        encoder_hidden_dims=(16,),
        denoiser_hidden_dims=(32,),
        diffusion=DiffusionConfig(timesteps=20),
    )


class PerturbDiffusionModelTest(unittest.TestCase):
    def test_training_losses_are_scalar_and_differentiable(self) -> None:
        torch.manual_seed(1)
        cfg = small_config()
        model = PerturbDiffusionModel(cfg)
        x0 = torch.randn(8, cfg.n_genes)
        perturbation_ids = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0])

        out = model.training_losses(x0, perturbation_ids)

        self.assertEqual(out["loss"].ndim, 0)
        self.assertEqual(out["prediction"].shape, x0.shape)
        out["loss"].backward()
        self.assertTrue(
            any(p.grad is not None for p in model.parameters() if p.requires_grad)
        )

    def test_control_embedding_is_zero_and_counterfactual_shape_matches(self) -> None:
        torch.manual_seed(2)
        cfg = small_config()
        model = PerturbDiffusionModel(cfg)
        x0 = torch.randn(4, cfg.n_genes)
        ids = torch.zeros(4, dtype=torch.long)

        condition = model.encode_condition(x0, ids)
        self.assertTrue(
            torch.allclose(condition["delta_base"], torch.zeros_like(condition["delta_base"]))
        )
        self.assertTrue(torch.allclose(condition["z_sem"], condition["z_pert"]))

        samples = model.sample_counterfactual(
            x_ctrl=x0[:2],
            target_ids=torch.tensor([1, 2]),
            steps=5,
        )
        self.assertEqual(samples.shape, x0[:2].shape)

    def test_counterfactual_uses_z_basal_plus_scaled_delta(self) -> None:
        torch.manual_seed(5)
        cfg = small_config()
        model = PerturbDiffusionModel(cfg)
        x0 = torch.randn(2, cfg.n_genes)
        target_ids = torch.tensor([1, 2], dtype=torch.long)
        guide_scales = torch.tensor([0.5, 2.0], dtype=torch.float32)

        with torch.no_grad():
            z_sem = model.encoder(x0)
            expected = model.perturbation.counterfactual_latent(
                z_sem,
                target_ids,
                source_ids=torch.zeros_like(target_ids),
                guide_scales=guide_scales,
            )
            baseline = z_sem + model.perturbation.embedding(target_ids) * guide_scales.unsqueeze(-1)

        self.assertTrue(torch.allclose(expected, baseline))

    def test_x0_prediction_type_runs_end_to_end(self) -> None:
        torch.manual_seed(3)
        cfg = ModelConfig(
            n_genes=20,
            n_perturbations=3,
            latent_dim=8,
            encoder_hidden_dims=(16,),
            denoiser_hidden_dims=(32,),
            diffusion=DiffusionConfig(timesteps=20, prediction_type="x0"),
        )
        model = PerturbDiffusionModel(cfg)
        x0 = torch.randn(8, cfg.n_genes)
        perturbation_ids = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0])

        out = model.training_losses(x0, perturbation_ids)
        self.assertEqual(out["loss"].ndim, 0)
        out["loss"].backward()

        samples = model.sample_counterfactual(
            x_ctrl=x0[:2],
            target_ids=torch.tensor([1, 2]),
            steps=5,
        )
        self.assertEqual(samples.shape, x0[:2].shape)

    def test_weighted_mse_and_signature_weights(self) -> None:
        weights = weighted_mse_loss(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            torch.tensor([[1.0, 2.0], [1.0, 1.0]]),
        )
        self.assertEqual(weights.ndim, 0)

        signature_weights = build_signature_gene_weights(
            ("A", "B", "C"),
            {"A": 1, "B": 2},
            summary_path=None,
            signature_gene_weight=2.0,
        )
        self.assertEqual(signature_weights.shape, (3, 3))
        self.assertTrue(torch.allclose(signature_weights, torch.ones_like(signature_weights)))

    def test_gears_autofocus_direction_loss_is_scalar(self) -> None:
        loss = gears_autofocus_direction_loss(
            torch.tensor([[2.0, 0.5], [0.0, 3.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 2.0]]),
            torch.ones(2, 2),
            control_mean=torch.tensor([1.0, 1.0]),
            gamma=2.0,
            direction_lambda=1e-3,
        )

        self.assertEqual(loss.ndim, 0)
        self.assertGreater(float(loss), 0.0)

    def test_gears_reconstruction_loss_runs_end_to_end(self) -> None:
        torch.manual_seed(4)
        cfg = ModelConfig(
            n_genes=20,
            n_perturbations=3,
            latent_dim=8,
            encoder_hidden_dims=(16,),
            denoiser_hidden_dims=(32,),
            diffusion=DiffusionConfig(timesteps=20, prediction_type="x0"),
            losses=TrainingLossConfig(reconstruction_loss="gears"),
        )
        model = PerturbDiffusionModel(cfg)
        x0 = torch.randn(8, cfg.n_genes)
        model.set_control_expression_mean(x0[:2].mean(dim=0))
        perturbation_ids = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0])

        out = model.training_losses(x0, perturbation_ids)

        self.assertEqual(out["loss"].ndim, 0)
        self.assertEqual(out["eval_mse_loss"].ndim, 0)
        out["loss"].backward()

    def test_direction_align_loss_prefers_same_target_cosine(self) -> None:
        torch.manual_seed(6)
        cfg = ModelConfig(
            n_genes=20,
            n_perturbations=3,
            latent_dim=8,
            encoder_hidden_dims=(16,),
            denoiser_hidden_dims=(32,),
            diffusion=DiffusionConfig(timesteps=20, prediction_type="x0"),
            losses=TrainingLossConfig(lambda_direction_align=1.0),
        )
        model = PerturbDiffusionModel(cfg)
        x0 = torch.randn(6, cfg.n_genes)
        perturbation_ids = torch.tensor([0, 1, 1, 2, 2, 0])
        target_ids = torch.tensor([0, 1, 1, 2, 2, 0])

        out = model.training_losses(x0, perturbation_ids, target_ids)

        self.assertEqual(out["direction_align_loss"].ndim, 0)
        self.assertGreaterEqual(float(out["direction_align_loss"]), 0.0)

    def test_predicted_activity_cnn_runs_inside_model(self) -> None:
        torch.manual_seed(7)
        feature_dim = guide_cnn_feature_dim(sequence_length=26)
        cfg = ModelConfig(
            n_genes=12,
            n_perturbations=2,
            latent_dim=4,
            encoder_hidden_dims=(8,),
            denoiser_hidden_dims=(16,),
            diffusion=DiffusionConfig(timesteps=20),
            guide_encoder=GuideFeatureEncoderConfig(
                enabled=True,
                feature_dim=feature_dim,
                activity_source="predicted",
                architecture="cnn",
                sequence_length=26,
                cnn_channels=(8, 8),
                cnn_kernel_size=3,
                cnn_dropout=0.1,
                lambda_strength=0.5,
            ),
        )
        model = PerturbDiffusionModel(cfg)
        x0 = torch.randn(3, cfg.n_genes)
        perturbation_ids = torch.tensor([0, 1, 2])
        guide_features = torch.randn(3, feature_dim)
        guide_activity_targets = torch.rand(3)

        out = model.training_losses(
            x0,
            perturbation_ids,
            target_ids=perturbation_ids,
            guide_features=guide_features,
            guide_activity_targets=guide_activity_targets,
        )

        self.assertEqual(out["loss"].ndim, 0)
        self.assertEqual(out["guide_strength_loss"].ndim, 0)
        self.assertIsNotNone(out["activity_embedding"])
        self.assertEqual(out["activity_embedding"].shape, (3, cfg.guide_encoder.activity_condition_dim))
        self.assertTrue(torch.allclose(out["activity_embedding"][0], torch.zeros_like(out["activity_embedding"][0])))
        out["loss"].backward()

        samples = model.sample_counterfactual(
            x_ctrl=x0[:2],
            target_ids=torch.tensor([1, 2]),
            guide_features=guide_features[:2],
            steps=5,
        )
        self.assertEqual(samples.shape, x0[:2].shape)

    def test_predicted_activity_is_explicit_denoiser_condition(self) -> None:
        torch.manual_seed(9)
        cfg = ModelConfig(
            n_genes=10,
            n_perturbations=2,
            latent_dim=4,
            encoder_hidden_dims=(8,),
            denoiser_hidden_dims=(16,),
            guide_encoder=GuideFeatureEncoderConfig(
                enabled=True,
                feature_dim=1,
                activity_source="predicted",
                explicit_condition=True,
                activity_condition_dim=3,
                architecture="mlp",
                hidden_dims=(),
                doser_type="linear",
            ),
        )
        model = PerturbDiffusionModel(cfg)
        self.assertEqual(model.denoiser.activity_condition_dim, 3)
        x0 = torch.randn(2, cfg.n_genes)
        perturbation_ids = torch.tensor([1, 1])
        guide_features = torch.tensor([[0.25], [1.0]])
        condition = model.encode_condition(
            x0,
            perturbation_ids,
            target_ids=perturbation_ids,
            guide_features=guide_features,
        )
        self.assertIsNotNone(condition["activity_embedding"])
        self.assertEqual(condition["activity_embedding"].shape, (2, 3))
        self.assertFalse(
            torch.allclose(condition["activity_embedding"][0], condition["activity_embedding"][1])
        )

        timesteps = torch.tensor([1, 1])
        x_t = torch.randn_like(x0)
        with_activity = model.denoiser(
            x_t,
            timesteps,
            condition["z_pert"],
            condition["activity_embedding"],
        )
        without_activity = model.denoiser(x_t, timesteps, condition["z_pert"])
        self.assertFalse(torch.allclose(with_activity, without_activity))

    def test_predicted_activity_does_not_change_target_delta_magnitude(self) -> None:
        torch.manual_seed(8)
        cfg = ModelConfig(
            n_genes=10,
            n_perturbations=2,
            latent_dim=4,
            encoder_hidden_dims=(8,),
            denoiser_hidden_dims=(16,),
            guide_encoder=GuideFeatureEncoderConfig(
                enabled=True,
                feature_dim=1,
                activity_source="predicted",
                architecture="mlp",
                hidden_dims=(),
                doser_type="linear",
            ),
        )
        model = PerturbDiffusionModel(cfg)
        with torch.no_grad():
            model.perturbation.embedding.weight[1].fill_(2.0)
            self.assertIsNotNone(model.guide_strength)
            linear = model.guide_strength.network[0]
            self.assertIsInstance(linear, torch.nn.Linear)
            linear.weight.fill_(1.0)
            linear.bias.zero_()

        x0 = torch.randn(2, cfg.n_genes)
        perturbation_ids = torch.tensor([1, 1])
        guide_features = torch.tensor([[0.25], [1.0]])

        condition = model.encode_condition(
            x0,
            perturbation_ids,
            target_ids=perturbation_ids,
            guide_features=guide_features,
        )

        self.assertTrue(torch.allclose(condition["delta"], condition["delta_base"]))
        self.assertTrue(torch.allclose(condition["delta"][0], condition["delta"][1]))
        self.assertIsNotNone(condition["activity_embedding"])
        self.assertFalse(
            torch.allclose(condition["activity_embedding"][0], condition["activity_embedding"][1])
        )

    def test_activity_delta_scaling_can_be_enabled_for_legacy_behavior(self) -> None:
        torch.manual_seed(10)
        cfg = ModelConfig(
            n_genes=10,
            n_perturbations=2,
            latent_dim=4,
            encoder_hidden_dims=(8,),
            denoiser_hidden_dims=(16,),
            guide_encoder=GuideFeatureEncoderConfig(
                enabled=True,
                feature_dim=1,
                activity_source="predicted",
                explicit_condition=False,
                activity_scales_delta=True,
                architecture="mlp",
                hidden_dims=(),
                doser_type="linear",
            ),
        )
        model = PerturbDiffusionModel(cfg)
        with torch.no_grad():
            model.perturbation.embedding.weight[1].fill_(2.0)
            self.assertIsNotNone(model.guide_strength)
            linear = model.guide_strength.network[0]
            self.assertIsInstance(linear, torch.nn.Linear)
            linear.weight.fill_(1.0)
            linear.bias.zero_()

        x0 = torch.randn(2, cfg.n_genes)
        perturbation_ids = torch.tensor([1, 1])
        guide_features = torch.tensor([[0.25], [1.0]])
        condition = model.encode_condition(
            x0,
            perturbation_ids,
            target_ids=perturbation_ids,
            guide_features=guide_features,
        )

        self.assertLess(
            float(condition["delta"][0].norm().detach()),
            float(condition["delta"][1].norm().detach()),
        )
        self.assertIsNone(condition["activity_embedding"])

    def test_direct_activity_mode_uses_scalar_features(self) -> None:
        torch.manual_seed(11)
        cfg = ModelConfig(
            n_genes=10,
            n_perturbations=2,
            latent_dim=4,
            encoder_hidden_dims=(8,),
            denoiser_hidden_dims=(16,),
            guide_encoder=GuideFeatureEncoderConfig(
                enabled=True,
                feature_dim=1,
                activity_source="direct",
                explicit_condition=True,
                activity_condition_dim=3,
                architecture="mlp",
                hidden_dims=(),
                doser_type="linear",
            ),
        )
        model = PerturbDiffusionModel(cfg)
        x0 = torch.randn(2, cfg.n_genes)
        perturbation_ids = torch.tensor([1, 1])
        guide_features = torch.tensor([[0.25], [1.0]])

        out = model.training_losses(
            x0,
            perturbation_ids,
            target_ids=perturbation_ids,
            guide_features=guide_features,
        )

        self.assertEqual(out["loss"].ndim, 0)
        self.assertEqual(out["activity_embedding"].shape, (2, 3))
        self.assertFalse(torch.allclose(out["activity_embedding"][0], out["activity_embedding"][1]))
        samples = model.sample_counterfactual(
            x_ctrl=x0[:2],
            target_ids=torch.tensor([1, 2]),
            guide_features=guide_features,
            steps=5,
        )
        self.assertEqual(samples.shape, x0[:2].shape)


if __name__ == "__main__":
    unittest.main()
