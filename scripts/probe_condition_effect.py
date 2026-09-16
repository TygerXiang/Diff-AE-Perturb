"""Sanity probe: does the conditional DDIM sampler actually respond to the condition?

Runs counterfactual generation from the same control input for several
perturbation ids and reports pairwise distances between the generated profiles,
the magnitude of the latent condition change, and the response of the denoiser
to the condition.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.config import ModelConfig  # noqa: E402
from perturbdiff.model import PerturbDiffusionModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    import anndata as ad

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    defaults = ModelConfig(n_genes=config.n_genes, n_perturbations=config.n_perturbations)
    missing = {
        f.name: getattr(defaults, f.name)
        for f in dataclasses.fields(ModelConfig)
        if not hasattr(config, f.name)
    }
    if missing:
        config = dataclasses.replace(config, **missing)
    model = PerturbDiffusionModel(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(args.device).eval()

    adata = ad.read_h5ad(args.data_path)
    labels = adata.obs["perturbation_label"].astype(str).to_numpy()
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = np.asarray(X, dtype=np.float32)
    control_mean = X[labels == args.control_label].mean(axis=0)

    mapping = checkpoint["perturbation_mapping"]
    reverse = {index: label for label, index in mapping.items()}
    control_tensor = torch.from_numpy(control_mean).unsqueeze(0).to(torch.device(args.device))

    probe_ids = [0, 1, 5, 8, 14, 23]
    outputs = {}
    with torch.no_grad():
        for perturb_id in probe_ids:
            ids = torch.tensor([perturb_id], device=args.device, dtype=torch.long)
            outputs[perturb_id] = model.sample_counterfactual(control_tensor, ids, steps=args.steps)[0].cpu().numpy()

    print("control_mean stats: min %.4f max %.4f mean %.4f" % (control_mean.min(), control_mean.max(), control_mean.mean()))
    for perturb_id, sample in outputs.items():
        delta = sample - control_mean
        print(
            "id %3d (%s): pred_mean %.4f, ||pred-control|| %.4f, max|delta| %.4f, n_genes|delta|>0.05: %d"
            % (
                perturb_id,
                reverse.get(perturb_id, "?"),
                sample.mean(),
                np.linalg.norm(delta),
                np.abs(delta).max(),
                int((np.abs(delta) > 0.05).sum()),
            )
        )

    print("\npairwise ||pred_i - pred_j||:")
    ids = list(outputs)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            print(f"  {left} vs {right}: {np.linalg.norm(outputs[left] - outputs[right]):.4f}")

    # Direct denoiser sensitivity to the condition.
    x_t = torch.from_numpy(control_mean).unsqueeze(0).to(torch.device(args.device))
    timesteps = torch.full((1,), 100, device=args.device, dtype=torch.long)
    z_control = model.encoder(x_t)
    with torch.no_grad():
        base = model.denoiser(x_t, timesteps, z_control)
        denoiser_rows = []
        for perturb_id in [1, 5, 14, 23]:
            delta = model.perturbation.embedding(torch.tensor([perturb_id], device=args.device))
            perturbed = model.denoiser(x_t, timesteps, z_control + delta)
            denoiser_rows.append(
                {
                    "perturbation_id": int(perturb_id),
                    "label": reverse.get(perturb_id, "?"),
                    "delta_l2": float(delta.norm().item()),
                    "dx0_l2": float((perturbed - base).norm().item()),
                }
            )
            print(
                "denoiser z_cond change id %d: ||dz|| %.4f -> ||dx0|| %.4f"
                % (perturb_id, delta.norm().item(), (perturbed - base).norm().item())
            )
    print("\nencoder z_sem control L2: %.4f" % z_control.norm().item())

    if args.json_out is not None:
        payload = {
            "checkpoint": str(args.checkpoint),
            "steps": args.steps,
            "control_mean_stats": {
                "min": float(control_mean.min()),
                "max": float(control_mean.max()),
                "mean": float(control_mean.mean()),
            },
            "z_sem_control_mean_expression_l2": float(z_control.norm().item()),
            "generated": [
                {
                    "perturbation_id": int(perturb_id),
                    "label": reverse.get(perturb_id, "?"),
                    "pred_mean": float(sample.mean()),
                    "pred_minus_control_l2": float(np.linalg.norm(sample - control_mean)),
                    "max_abs_delta": float(np.abs(sample - control_mean).max()),
                    "n_genes_abs_delta_gt_0.05": int((np.abs(sample - control_mean) > 0.05).sum()),
                }
                for perturb_id, sample in outputs.items()
            ],
            "pairwise_generated_l2": {
                f"{left}_vs_{right}": float(np.linalg.norm(outputs[left] - outputs[right]))
                for index, left in enumerate(ids)
                for right in ids[index + 1 :]
            },
            "denoiser_condition_sensitivity": denoiser_rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
