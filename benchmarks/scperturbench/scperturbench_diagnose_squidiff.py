"""Diagnose the Squidiff sampling scale.

Loads the checkpoint produced by the Squidiff adapter and compares the sampled cell
statistics against the training data under different (a) conditioning vectors and
(b) DDIM respacings. This separates "conditioning is off-distribution" from
"sampling schedule / training schedule mismatch".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import anndata as ad
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True, help="<out>/squidiff directory")
    parser.add_argument("--repo", default="/root/Squidiff")
    parser.add_argument("--n-cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument("--clamp", action="store_true", default=False, help="apply non-negativity denoised_fn")
    args = parser.parse_args()
    denoised_fn = (lambda x: x.clamp(min=0.0)) if args.clamp else None

    repo_root = Path(args.repo)
    if (repo_root / "Squidiff" / "script_util.py").exists():
        sys.path.insert(0, str(repo_root))
    from Squidiff.script_util import (
        args_to_dict,
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    train = ad.read_h5ad(args.experiment / "train.h5ad")
    X = np.asarray(train.X, dtype=np.float32)
    print(f"train X: mean={X.mean():.4f} std={X.std():.4f} max={X.max():.4f}")
    genes = train.n_vars

    device = torch.device(args.device)

    # --- checkpoint health check: does the saved model actually predict the training noise?
    model_args = dict(model_and_diffusion_defaults())
    model_args.update(
        {
            "gene_size": genes,
            "output_dim": genes,
            "num_layers": 3,
            "class_cond": False,
            "use_encoder": True,
            "use_drug_structure": False,
            "diffusion_steps": 1000,
            "timestep_respacing": "ddim50",
            "dropout": 0.0,
        }
    )
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(model_args, model_and_diffusion_defaults().keys())
    )
    state = torch.load(args.experiment / "checkpoint" / "model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:5]}")
    model.to(device)
    model.eval()

    x_start = torch.from_numpy(X[:64]).to(device)
    torch.manual_seed(0)
    for timestep in [0, 10, 25, 49]:
        t = torch.full((x_start.shape[0],), timestep, device=device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        losses = diffusion.training_losses(
            model,
            x_start,
            t,
            model_kwargs={
                "x_start": x_start,
                "group": torch.zeros(x_start.shape[0], device=device),
                "drug_dose": None,
                "control_feature": None,
            },
            noise=noise,
        )
        print(f"  training_loss at sampled t={timestep}: mse={losses['mse'].mean().item():.4f}")
        # eps prediction quality
        x_t = diffusion.q_sample(x_start, t, noise=noise)
        with torch.no_grad():
            eps_pred = model(
                x_t,
                diffusion._scale_timesteps(t),
                x_start=x_start,
                group=None,
                drug_dose=None,
                control_feature=None,
            )
        corr = float(
            np.corrcoef(eps_pred.flatten().cpu().numpy(), noise.flatten().cpu().numpy())[0, 1]
        )
        print(
            f"    eps_pred: std={eps_pred.std().item():.4f} corr_with_true_noise={corr:.4f} "
            f"x_t std={x_t.std().item():.4f}"
        )
    del model, diffusion
    torch.cuda.empty_cache()

    for respacing in ["ddim50", ""]:
        if args.trace:
            model_args = dict(model_and_diffusion_defaults())
            model_args.update(
                {
                    "gene_size": genes,
                    "output_dim": genes,
                    "num_layers": 3,
                    "class_cond": False,
                    "use_encoder": True,
                    "use_drug_structure": False,
                    "diffusion_steps": 1000,
                    "timestep_respacing": respacing,
                    "dropout": 0.0,
                }
            )
            model, diffusion = create_model_and_diffusion(
                **args_to_dict(model_args, model_and_diffusion_defaults().keys())
            )
            state = torch.load(args.experiment / "checkpoint" / "model.pt", map_location="cpu")
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            groups = train.obs["Group"].to_numpy()
            cells = torch.from_numpy(X[groups != 0][: args.n_cells]).to(device)
            with torch.no_grad():
                z = model.encoder(cells, label=None, drug_dose=None, control_feature=None)
                shape = (args.n_cells, genes)
                sample = torch.randn(shape, device=device)
                indices = list(range(diffusion.num_timesteps))[::-1]
                watch = {indices[0], indices[1], indices[2], indices[len(indices) // 2], indices[-3], indices[-2], indices[-1]}
                print(f"--- trace respacing={respacing!r} start |x|={sample.abs().mean().item():.4f}")
                for i in indices:
                    t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                    out = diffusion.ddim_sample(
                        model, sample, t, denoised_fn=denoised_fn, model_kwargs={"z_mod": z}, eta=0.0
                    )
                    sample = out["sample"]
                    if i in watch:
                        print(
                            f"    i={i:4d} |x|={sample.abs().mean().item():9.4f} "
                            f"pred_xstart|mean|={out['pred_xstart'].abs().mean().item():9.4f} "
                            f"sqrt_a={float(diffusion.sqrt_alphas_cumprod[i]):.4f}"
                        )
            del model, diffusion
            torch.cuda.empty_cache()
            continue
        for ckpt_name in ["model.pt", "model_0.9999.pt"]:
            ckpt_path = args.experiment / "checkpoint" / ckpt_name
            if not ckpt_path.exists():
                continue
            model_args = dict(model_and_diffusion_defaults())
            model_args.update(
                {
                    "gene_size": genes,
                    "output_dim": genes,
                    "num_layers": 3,
                    "class_cond": False,
                    "use_encoder": True,
                    "use_drug_structure": False,
                    "diffusion_steps": 1000,
                    "timestep_respacing": respacing,
                    "dropout": 0.0,
                }
            )
            model, diffusion = create_model_and_diffusion(
                **args_to_dict(model_args, model_and_diffusion_defaults().keys())
            )
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            print(f"--- respacing={respacing!r} ckpt={ckpt_name} num_timesteps={diffusion.num_timesteps}")

            groups = train.obs["Group"].to_numpy()
            for label, name in [(0.0, "control"), (None, "perturbed")]:
                if label is None:
                    mask = groups != 0
                else:
                    mask = groups == label
                cells = torch.from_numpy(X[mask][:64]).to(device)
                with torch.no_grad():
                    z = model.encoder(
                        cells, label=None, drug_dose=None, control_feature=None
                    )
                    print(f"    z_sem[{name}]: mean={z.mean().item():.3f} std={z.std().item():.3f} absmax={z.abs().max().item():.3f}")
                    samples = diffusion.ddim_sample_loop(
                        model,
                        shape=(args.n_cells, genes),
                        denoised_fn=denoised_fn,
                        model_kwargs={"z_mod": z[: args.n_cells]},
                        device=device,
                    )
                    arr = samples.detach().cpu().numpy()
                print(
                    f"    sample[{name}]: mean={arr.mean():.4f} std={arr.std():.4f} "
                    f"max={arr.max():.4f} min={arr.min():.4f}"
                )
            del model, diffusion
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


