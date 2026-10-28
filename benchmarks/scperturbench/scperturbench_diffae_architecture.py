"""Architecture diagnostic for the Diff-AE-Perturb GSE132080 checkpoint.

The benchmark showed this checkpoint has delta-correlation ~0.03 while GEARS/scGen/linear sit at
0.74-0.90.  This script tests the specific hypothesis that the denoiser's conditioning pathway
leaks the answer during training, so the learned per-perturbation ``delta`` is never used and the
inference-time counterfactual conditioning carries no condition information.

Experiments
-----------
A. eps-leakage: same batch of real perturbed cells, four conditioning variants
   (``self`` = encoder(x0) as in training, ``permuted`` = encoder of a different cell,
   ``zero``, ``control_mean``).  If ``self`` is far better, the encoder carries x0.
B. magnitude: per-condition true latent shift ``enc(condition) - enc(control)`` vs the learned
   ``delta`` row, and the cell-to-cell spread of the control encoding (the noise floor).
C. inference sensitivity: predicted pseudobulk spread across conditions vs the true spread and
   vs the spread induced by swapping control cells.
D. delta-scale sweep: how much does the prediction move when delta is multiplied by 0/1/10/100?

Run inside the container with the repo mounted at /root/Diff-AE-Perturb.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/Diff-AE-Perturb/src")


def to_dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--test-h5ad", type=Path, required=True)
    parser.add_argument("--condition-col", default="perturbation_label")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--n-cells", type=int, default=512)
    parser.add_argument("--n-control", type=int, default=256)
    parser.add_argument("--n-pred-control", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import anndata as ad
    from dataclasses import replace

    from perturbdiff.config import GuideFeatureEncoderConfig
    from perturbdiff.model import PerturbDiffusionModel

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    payload = torch.load(args.checkpoint, map_location="cpu")
    # Same patch the project's own evaluate_benchmark.load_model applies: this checkpoint was
    # written before GuideFeatureEncoderConfig existed, so the field has to be injected.
    cfg = payload["model_config"]
    if not hasattr(cfg, "guide_encoder"):
        cfg = replace(cfg, guide_encoder=GuideFeatureEncoderConfig())
    mapping = {str(k): int(v) for k, v in payload["perturbation_mapping"].items()}
    model = PerturbDiffusionModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    denoiser = model.denoiser
    diffusion = model.diffusion
    encoder = model.encoder

    train = ad.read_h5ad(args.train_h5ad)
    test = ad.read_h5ad(args.test_h5ad)
    genes = [str(g) for g in train.var_names]
    if genes != [str(g) for g in test.var_names]:
        raise SystemExit("train/test gene order mismatch")
    X = to_dense(train.X).astype(np.float32)
    labels = train.obs[args.condition_col].astype(str).to_numpy()
    test_labels = test.obs[args.condition_col].astype(str).to_numpy()
    X_test = to_dense(test.X).astype(np.float32)

    report: dict = {
        "checkpoint": str(args.checkpoint),
        "n_genes": len(genes),
        "latent_dim": int(cfg.latent_dim),
        "n_perturbations": int(cfg.n_perturbations),
        "guide_encoder_enabled": bool(cfg.guide_encoder.enabled),
        "denoiser_input_dim": int(denoiser.network[0].in_features),
        "expected_input_dim_without_activity": int(cfg.n_genes + cfg.latent_dim + (cfg.latent_dim or cfg.latent_dim)),
        "lambda_semantic_align": float(cfg.losses.lambda_semantic_align),
        "lambda_direction_align": float(cfg.losses.lambda_direction_align),
        "reconstruction_loss": str(cfg.losses.reconstruction_loss),
        "prediction_type": str(diffusion.prediction_type),
    }

    control_mask = labels == args.control_label
    control_cells = X[control_mask]
    if control_cells.shape[0] > args.n_control:
        control_cells = control_cells[rng.choice(control_cells.shape[0], args.n_control, replace=False)]

    with torch.no_grad():
        # ---------------------------------------------------------------- A
        pick = rng.choice(X.shape[0], min(args.n_cells, X.shape[0]), replace=False)
        x0 = torch.from_numpy(X[pick])
        perm = torch.from_numpy(rng.permutation(x0.shape[0]))
        z_self = encoder(x0)
        z_perm = encoder(x0[perm])
        z_zero = torch.zeros_like(z_self)
        z_ctrl = encoder(torch.from_numpy(control_cells)).mean(dim=0, keepdim=True).expand_as(z_self)

        variants = {"self": z_self, "permuted": z_perm, "zero": z_zero, "control_mean": z_ctrl}
        buckets = [(1, 200), (200, 400), (400, 600), (600, 800), (800, 1001)]
        leak = {"prediction_type": str(diffusion.prediction_type), "buckets": []}
        for low, high in buckets:
            t = torch.randint(low, high, (x0.shape[0],))
            x_t, noise = diffusion.q_sample(x0, t)
            entry = {"t_low": low, "t_high": high, "variants": {}}
            for name, z in variants.items():
                raw = denoiser(x_t, t, z)
                x0_hat, eps_hat = diffusion.prediction_to_x0_and_eps(x_t, t, raw)
                entry["variants"][name] = {
                    "eps_mse": float(((eps_hat - noise) ** 2).mean()),
                    "eps_corr": pearson(eps_hat.numpy(), noise.numpy()),
                    "x0_mse": float(((x0_hat - x0) ** 2).mean()),
                    "x0_corr": pearson(x0_hat.numpy(), x0.numpy()),
                }
            leak["buckets"].append(entry)
        leak["reference_x0_mse_of_control_mean"] = float(
            ((torch.from_numpy(control_cells).mean(dim=0, keepdim=True) - x0) ** 2).mean()
        )
        leak["reference_x0_mse_of_zero"] = float((x0 ** 2).mean())
        report["A_conditioning_information"] = leak

        # ---------------------------------------------------------------- B
        conditions = sorted(
            {str(v) for v in test_labels if str(v) not in {args.control_label, "*"}}
        )
        z_control_mean = encoder(torch.from_numpy(control_cells)).mean(dim=0)
        control_spread = (encoder(torch.from_numpy(control_cells)) - z_control_mean).norm(dim=1)
        per_condition = []
        for condition in conditions:
            mask = labels == condition
            if not mask.any():
                continue
            cells = torch.from_numpy(X[mask])
            z_cells = encoder(cells)
            z_mean = z_cells.mean(dim=0)
            delta_real = z_mean - z_control_mean
            pidx = mapping.get(condition)
            delta_learned = (
                model.perturbation.embedding.weight[pidx].detach() if pidx is not None else None
            )
            entry = {
                "condition": condition,
                "train_cells": int(mask.sum()),
                "perturbation_id": pidx,
                "norm_delta_real": float(delta_real.norm()),
                "norm_z_control_mean": float(z_control_mean.norm()),
                "within_condition_spread": float((z_cells - z_mean).norm(dim=1).mean()),
            }
            if delta_learned is not None:
                entry["norm_delta_learned"] = float(delta_learned.norm())
                entry["cos_delta_real_vs_learned"] = float(
                    torch.nn.functional.cosine_similarity(
                        delta_real.unsqueeze(0), delta_learned.unsqueeze(0)
                    ).item()
                )
                entry["ratio_learned_over_real"] = float(
                    delta_learned.norm() / max(delta_real.norm(), 1e-12)
                )
            per_condition.append(entry)
        report["B_latent_geometry"] = {
            "control_cell_spread_norm_mean": float(control_spread.mean()),
            "per_condition": per_condition,
        }

        # ---------------------------------------------------------------- C + D
        ctrl = torch.from_numpy(control_cells[: args.n_pred_control])
        z_ctrl_cells = encoder(ctrl)
        print(
            f"[debug] X={X.shape} control_cells={control_cells.shape} ctrl={tuple(ctrl.shape)} "
            f"z_ctrl_cells={tuple(z_ctrl_cells.shape)} denoiser_in={denoiser.network[0].in_features}",
            flush=True,
        )
        sensitivity = {"per_condition": [], "delta_scale_sweep": []}
        pseudobulk = {}
        for condition in conditions[:8]:
            pidx = mapping.get(condition)
            if pidx is None:
                continue
            delta = model.perturbation.embedding.weight[pidx].detach()
            pred = diffusion.ddim_sample(
                denoiser,
                shape=ctrl.shape,
                z_cond=z_ctrl_cells + delta,
                steps=args.steps,
            )
            pseudobulk[condition] = pred.mean(dim=0).numpy()
            mask = test_labels == condition
            true_mean = X_test[mask].mean(axis=0) if mask.any() else None
            sensitivity["per_condition"].append(
                {
                    "condition": condition,
                    "pred_mean": float(pred.mean()),
                    "pred_max": float(pred.max()),
                    "true_mean": float(true_mean.mean()) if true_mean is not None else None,
                }
            )
        if len(pseudobulk) >= 2:
            keys = list(pseudobulk)
            stack_pred = np.stack([pseudobulk[k] for k in keys])
            stack_true = np.stack(
                [X_test[test_labels == k].mean(axis=0) for k in keys if (test_labels == k).any()]
            )
            sensitivity["between_condition_sd_pred"] = float(stack_pred.std(axis=0).mean())
            sensitivity["between_condition_sd_true"] = float(stack_true.std(axis=0).mean())
        # noise floor: same condition, different control cells
        if pseudobulk:
            key = keys[0]
            repeats = []
            for rep in range(3):
                idx = rng.choice(control_cells.shape[0], args.n_pred_control, replace=False)
                sub = encoder(torch.from_numpy(control_cells[idx]))
                pred = diffusion.ddim_sample(
                    denoiser,
                    shape=(sub.shape[0], X.shape[1]),
                    z_cond=sub + model.perturbation.embedding.weight[mapping[key]].detach(),
                    steps=args.steps,
                )
                repeats.append(pred.mean(dim=0).numpy())
            sensitivity["control_swap_sd_pred"] = float(np.stack(repeats).std(axis=0).mean())

        # delta scale sweep on the first available condition
        for condition in conditions:
            pidx = mapping.get(condition)
            if pidx is None:
                continue
            delta = model.perturbation.embedding.weight[pidx].detach()
            base = None
            for scale in (0.0, 1.0, 10.0, 100.0, 1000.0):
                pred = diffusion.ddim_sample(
                    denoiser,
                    shape=ctrl.shape,
                    z_cond=z_ctrl_cells + scale * delta,
                    steps=args.steps,
                ).mean(dim=0)
                if base is None:
                    base = pred
                sensitivity["delta_scale_sweep"].append(
                    {
                        "condition": condition,
                        "scale": scale,
                        "mean": float(pred.mean()),
                        "max": float(pred.max()),
                        "l2_from_scale0": float((pred - base).norm()),
                    }
                )
            break
        report["C_inference_sensitivity"] = sensitivity

        # ---------------------------------------------------------------- E
        # The decisive test: same denoiser, two conditioning *interfaces*.
        #   training interface : z = encoder(cells actually under that perturbation)
        #   production interface: z = encoder(control cells) + delta_target   (what predict_model does)
        interface = []
        control_pseudobulk = control_cells.mean(axis=0)
        for condition in conditions:
            pidx = mapping.get(condition)
            mask = test_labels == condition
            if pidx is None or not mask.any():
                continue
            cells = X_test[mask]
            take = int(min(cells.shape[0], args.n_pred_control))
            true_mean = cells.mean(axis=0)
            z_true = encoder(torch.from_numpy(cells[:take]))
            z_prod = encoder(ctrl) + model.perturbation.embedding.weight[pidx].detach()
            pred_true = diffusion.ddim_sample(
                denoiser, shape=(take, X.shape[1]), z_cond=z_true, steps=args.steps
            ).mean(dim=0).numpy()
            pred_prod = diffusion.ddim_sample(
                denoiser, shape=(ctrl.shape[0], X.shape[1]), z_cond=z_prod, steps=args.steps
            ).mean(dim=0).numpy()
            interface.append(
                {
                    "condition": condition,
                    "pearson_true_vs_encoder_of_true_cells": pearson(true_mean, pred_true),
                    "mse_true_vs_encoder_of_true_cells": float(((true_mean - pred_true) ** 2).mean()),
                    "pearson_true_vs_control_plus_delta": pearson(true_mean, pred_prod),
                    "mse_true_vs_control_plus_delta": float(((true_mean - pred_prod) ** 2).mean()),
                    "pearson_true_vs_control_pseudobulk": pearson(true_mean, control_pseudobulk),
                    "mse_true_vs_control_pseudobulk": float(((true_mean - control_pseudobulk) ** 2).mean()),
                    # delta correlation is the quantity the benchmark's pcc_delta measures;
                    # absolute pseudobulk Pearson is saturated near 1 by the shared baseline profile.
                    "delta_corr_true_vs_encoder_of_true_cells": pearson(
                        true_mean - control_pseudobulk, pred_true - control_pseudobulk
                    ),
                    "delta_corr_true_vs_control_plus_delta": pearson(
                        true_mean - control_pseudobulk, pred_prod - control_pseudobulk
                    ),
                    "mean_true": float(true_mean.mean()),
                    "mean_pred_via_encoder_of_true_cells": float(pred_true.mean()),
                    "mean_pred_via_control_plus_delta": float(pred_prod.mean()),
                    "l2_between_the_two_predictions": float(np.linalg.norm(pred_true - pred_prod)),
                }
            )
        report["E_conditioning_interface"] = interface

        # ---------------------------------------------------------------- F
        # Was the delta table ever updated from its initialisation?  The init
        # (a8c355f PerturbationModule.initialize_from_latent_shift) copies
        # mean(z_sem | target) - mean(z_sem | control) computed with whatever encoder existed at
        # that moment, and the table never enters the denoiser, so only a scale-invariant cosine
        # term can touch it afterwards (AdamW weight_decay=0.0 cannot shrink it either).
        torch.manual_seed(args.seed)
        fresh = PerturbDiffusionModel(cfg)
        init_rows = []
        with torch.no_grad():
            fresh_z_control = fresh.encoder(torch.from_numpy(control_cells)).mean(dim=0)
            for condition in conditions:
                mask = labels == condition
                pidx = mapping.get(condition)
                if pidx is None or not mask.any():
                    continue
                sub = X[mask]
                if sub.shape[0] > 400:
                    sub = sub[rng.choice(sub.shape[0], 400, replace=False)]
                delta_init = fresh.encoder(torch.from_numpy(sub)).mean(dim=0) - fresh_z_control
                delta_checkpoint = model.perturbation.embedding.weight[pidx].detach()
                init_rows.append(
                    {
                        "condition": condition,
                        "norm_delta_from_fresh_encoder": float(delta_init.norm()),
                        "norm_delta_in_checkpoint": float(delta_checkpoint.norm()),
                        "cos_fresh_init_vs_checkpoint": float(
                            torch.nn.functional.cosine_similarity(
                                delta_init.unsqueeze(0), delta_checkpoint.unsqueeze(0)
                            ).item()
                        ),
                    }
                )
        report["F_delta_initialisation"] = init_rows

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
