from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff import DiffusionConfig, ModelConfig, PerturbDiffusionModel


def main() -> None:
    torch.manual_seed(7)
    cfg = ModelConfig(
        n_genes=64,
        n_perturbations=5,
        latent_dim=16,
        encoder_hidden_dims=(32,),
        denoiser_hidden_dims=(64,),
        diffusion=DiffusionConfig(timesteps=50, prediction_type="x0"),
    )
    model = PerturbDiffusionModel(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x0 = torch.randn(16, cfg.n_genes)
    perturbation_ids = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4, 5])

    with torch.no_grad():
        z_sem = model.encoder(x0)
        model.initialize_perturbation_from_latent_shift(z_sem, perturbation_ids)

    out = model.training_losses(x0, perturbation_ids)
    out["loss"].backward()
    optim.step()

    samples = model.sample_counterfactual(
        x_ctrl=x0[:3],
        target_ids=torch.tensor([1, 2, 3]),
        steps=10,
    )
    print(f"loss={out['loss'].item():.4f}")
    print(f"sample_shape={tuple(samples.shape)}")


if __name__ == "__main__":
    main()
