from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import torch
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.data import CONTROL_PATTERNS
from perturbdiff.model import PerturbDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare guide-specific generation accuracy between target-level and "
            "sgRNA-level PerturbDiff checkpoints on the same test split."
        )
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "splits" / "gse132080_test.h5ad",
    )
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--sgrna-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--guide-col", default="guide_identity")
    parser.add_argument("--target-col", default="perturbation_label")
    parser.add_argument("--control-label", default="control")
    parser.add_argument(
        "--min-cells-thresholds",
        default="1,5,10",
        help="Comma-separated per-guide minimum test-cell thresholds used for summary aggregation.",
    )
    parser.add_argument("--control-cell-limit", type=int, default=None)
    parser.add_argument("--max-guides", type=int, default=None, help="Optional smoke-test limit on evaluated guides.")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def vector_mean(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x).mean(axis=0)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    if true_std == 0.0 or pred_std == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def evaluate_mean_profile(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r = pearson_r(y_true, y_pred)
    return {
        "pearson_r": r,
        "pearson_r2": float(r * r) if np.isfinite(r) else float("nan"),
        "mse": mse(y_true, y_pred),
    }


def is_control_guide(label: str) -> bool:
    lowered = str(label).strip().lower()
    return any(pattern.lower() in lowered for pattern in CONTROL_PATTERNS)


def average_metric(rows: list[dict[str, object]], method_key: str, metric_key: str) -> float:
    values: list[float] = []
    for row in rows:
        value = float(row[method_key][metric_key])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def generate_mean_profile(
    model: PerturbDiffusionModel,
    x_ctrl: torch.Tensor,
    *,
    perturbation_id: int,
    steps: int,
) -> np.ndarray:
    target_ids = torch.full(
        (x_ctrl.shape[0],),
        int(perturbation_id),
        dtype=torch.long,
        device=x_ctrl.device,
    )
    pred = model.sample_counterfactual(
        x_ctrl=x_ctrl,
        target_ids=target_ids,
        steps=steps,
    )
    return pred.mean(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)


def parse_thresholds(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("At least one min-cells threshold is required")
    return sorted(set(out))


def build_summary(
    rows: list[dict[str, object]],
    *,
    thresholds: list[int],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds:
        subset = [row for row in rows if int(row["n_test_cells"]) >= threshold]
        label = "all_guides" if threshold <= 1 else f"guides_ge_{threshold}_cells"
        target_r2 = average_metric(subset, "target_level", "pearson_r2")
        sgrna_r2 = average_metric(subset, "sgrna_level", "pearson_r2")
        target_mse = average_metric(subset, "target_level", "mse")
        sgrna_mse = average_metric(subset, "sgrna_level", "mse")
        summary[label] = {
            "n_guides": len(subset),
            "target_level_mean_pearson_r2": target_r2,
            "sgrna_level_mean_pearson_r2": sgrna_r2,
            "sgrna_minus_target_mean_pearson_r2": (
                float(sgrna_r2 - target_r2)
                if np.isfinite(target_r2) and np.isfinite(sgrna_r2)
                else float("nan")
            ),
            "target_level_mean_mse": target_mse,
            "sgrna_level_mean_mse": sgrna_mse,
            "target_minus_sgrna_mean_mse": (
                float(target_mse - sgrna_mse)
                if np.isfinite(target_mse) and np.isfinite(sgrna_mse)
                else float("nan")
            ),
            "sgrna_better_r2_guides": int(
                sum(
                    float(row["sgrna_level"]["pearson_r2"]) > float(row["target_level"]["pearson_r2"])
                    for row in subset
                    if np.isfinite(float(row["sgrna_level"]["pearson_r2"]))
                    and np.isfinite(float(row["target_level"]["pearson_r2"]))
                )
            ),
            "sgrna_lower_mse_guides": int(
                sum(
                    float(row["sgrna_level"]["mse"]) < float(row["target_level"]["mse"])
                    for row in subset
                    if np.isfinite(float(row["sgrna_level"]["mse"]))
                    and np.isfinite(float(row["target_level"]["mse"]))
                )
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.min_cells_thresholds)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable in this interpreter; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    adata = ad.read_h5ad(args.test_path)
    for col in (args.guide_col, args.target_col):
        if col not in adata.obs:
            raise KeyError(f"Column {col!r} not found in test AnnData")

    guide_labels = adata.obs[args.guide_col].astype(str).to_numpy()
    target_labels = adata.obs[args.target_col].astype(str).to_numpy()
    gene_names = np.asarray(adata.var_names.astype(str))
    x_np = to_dense(adata.X).astype(np.float32, copy=False)

    control_mask = target_labels == args.control_label
    if int(control_mask.sum()) == 0:
        raise ValueError(f"No control cells found with target label {args.control_label!r}")

    guide_mask = (~control_mask) & (guide_labels != "*") & np.array([not is_control_guide(label) for label in guide_labels])
    eval_guides = sorted(set(guide_labels[guide_mask]))
    if args.max_guides is not None:
        eval_guides = eval_guides[: max(0, int(args.max_guides))]
    if not eval_guides:
        raise ValueError("No guide labels selected for evaluation")

    control_indices = np.flatnonzero(control_mask)
    if args.control_cell_limit is not None and control_indices.size > args.control_cell_limit:
        rng = np.random.default_rng(args.seed)
        control_indices = np.sort(
            rng.choice(control_indices, size=int(args.control_cell_limit), replace=False)
        )
    x_ctrl = torch.from_numpy(x_np[control_indices]).to(device)

    target_state = torch.load(args.target_checkpoint, map_location=device, weights_only=False)
    target_model = PerturbDiffusionModel(target_state["model_config"]).to(device)
    target_model.load_state_dict(target_state["model_state_dict"])
    target_model.eval()
    target_mapping = {str(key): int(value) for key, value in target_state["perturbation_mapping"].items()}

    sgrna_state = torch.load(args.sgrna_checkpoint, map_location=device, weights_only=False)
    sgrna_model = PerturbDiffusionModel(sgrna_state["model_config"]).to(device)
    sgrna_model.load_state_dict(sgrna_state["model_state_dict"])
    sgrna_model.eval()
    sgrna_mapping = {str(key): int(value) for key, value in sgrna_state["perturbation_mapping"].items()}

    cached_target_preds: dict[str, np.ndarray] = {}
    cached_sgrna_preds: dict[str, np.ndarray] = {}
    per_guide: list[dict[str, object]] = []
    skipped_guides: list[dict[str, str]] = []

    for guide in eval_guides:
        mask = guide_labels == guide
        n_test_cells = int(mask.sum())
        targets = sorted(set(target_labels[mask]))
        if len(targets) != 1:
            raise ValueError(f"Guide {guide!r} maps to multiple target labels in test split: {targets}")
        target = targets[0]

        if target not in target_mapping:
            skipped_guides.append({"guide": guide, "reason": f"target {target!r} missing from target checkpoint mapping"})
            continue
        if guide not in sgrna_mapping:
            skipped_guides.append({"guide": guide, "reason": "guide missing from sgRNA checkpoint mapping"})
            continue

        y_true_mean = vector_mean(adata[mask].X).astype(np.float64, copy=False)

        if target not in cached_target_preds:
            cached_target_preds[target] = generate_mean_profile(
                target_model,
                x_ctrl,
                perturbation_id=target_mapping[target],
                steps=args.ddim_steps,
            )
        if guide not in cached_sgrna_preds:
            cached_sgrna_preds[guide] = generate_mean_profile(
                sgrna_model,
                x_ctrl,
                perturbation_id=sgrna_mapping[guide],
                steps=args.ddim_steps,
            )

        y_target_pred = cached_target_preds[target]
        y_sgrna_pred = cached_sgrna_preds[guide]
        target_gene_idx = np.flatnonzero(gene_names == target)
        target_gene_index = int(target_gene_idx[0]) if target_gene_idx.size else None

        row = {
            "guide": guide,
            "target": target,
            "n_test_cells": n_test_cells,
            "target_level": evaluate_mean_profile(y_true_mean, y_target_pred),
            "sgrna_level": evaluate_mean_profile(y_true_mean, y_sgrna_pred),
            "sgrna_minus_target_r2": float(
                evaluate_mean_profile(y_true_mean, y_sgrna_pred)["pearson_r2"]
                - evaluate_mean_profile(y_true_mean, y_target_pred)["pearson_r2"]
            ),
            "target_minus_sgrna_mse": float(
                evaluate_mean_profile(y_true_mean, y_target_pred)["mse"]
                - evaluate_mean_profile(y_true_mean, y_sgrna_pred)["mse"]
            ),
            "target_gene_index": target_gene_index,
            "target_gene_true_mean": None if target_gene_index is None else float(y_true_mean[target_gene_index]),
            "target_gene_target_level_mean": None if target_gene_index is None else float(y_target_pred[target_gene_index]),
            "target_gene_sgrna_level_mean": None if target_gene_index is None else float(y_sgrna_pred[target_gene_index]),
        }
        per_guide.append(row)
        print(
            f"{guide}: "
            f"target R2={row['target_level']['pearson_r2']:.4f}, "
            f"sgRNA R2={row['sgrna_level']['pearson_r2']:.4f}, "
            f"target MSE={row['target_level']['mse']:.4f}, "
            f"sgRNA MSE={row['sgrna_level']['mse']:.4f}"
        )

    summary = build_summary(per_guide, thresholds=thresholds)
    out = {
        "config": {
            "test_path": str(args.test_path),
            "target_checkpoint": str(args.target_checkpoint),
            "sgrna_checkpoint": str(args.sgrna_checkpoint),
            "guide_col": args.guide_col,
            "target_col": args.target_col,
            "control_label": args.control_label,
            "control_cells_used": int(x_ctrl.shape[0]),
            "ddim_steps": args.ddim_steps,
            "device": str(device),
            "thresholds": thresholds,
            "max_guides": args.max_guides,
        },
        "summary": summary,
        "per_guide": per_guide,
        "skipped_guides": skipped_guides,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved comparison to {args.output_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
