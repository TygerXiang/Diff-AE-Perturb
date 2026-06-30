from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Replogle in STATE/PerturbDiff style: HVG2000 + divide-by-10, "
            "then split with held-out cell line and 30/70 perturbation partition."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=Path(
            r"C:\Users\Admin\myprojects\PerturbDiff\perturb_data\finetune_data\nadig_processed_data\replogle.h5ad"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "replogle_state_style",
    )
    parser.add_argument("--cell-line-col", default="cell_line")
    parser.add_argument("--perturbation-col", default="gene")
    parser.add_argument("--holdout-cell-line", default="hepg2")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--holdout-train-pert-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _divide_by_ten(x):
    if sparse.issparse(x):
        y = x.copy().astype(np.float32)
        y.data /= 10.0
        return y
    return np.asarray(x, dtype=np.float32) / 10.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.input_h5ad)
    if args.cell_line_col not in adata.obs:
        raise KeyError(f"Missing column: {args.cell_line_col}")
    if args.perturbation_col not in adata.obs:
        raise KeyError(f"Missing column: {args.perturbation_col}")
    if "highly_variable" not in adata.var:
        raise KeyError("Expected adata.var['highly_variable'] for HVG selection.")

    # A.2.1/A.2.2 style: use HVG2000 then scale log-expression by /10.
    hvg_mask = adata.var["highly_variable"].to_numpy().astype(bool)
    if int(hvg_mask.sum()) != 2000:
        # Be permissive but explicit.
        print(f"[warn] highly_variable count is {int(hvg_mask.sum())}, not 2000.")
    adata = adata[:, hvg_mask].copy()
    adata.X = _divide_by_ten(adata.X)

    obs = adata.obs.copy()
    cell_line = obs[args.cell_line_col].astype(str).to_numpy()
    pert = obs[args.perturbation_col].astype(str).to_numpy()

    holdout_mask = cell_line == args.holdout_cell_line
    non_holdout_mask = ~holdout_mask
    holdout_pert_mask = holdout_mask & (pert != args.control_label)

    holdout_perturbations = np.unique(pert[holdout_pert_mask])
    rng = np.random.default_rng(args.seed)
    shuffled = holdout_perturbations.copy()
    rng.shuffle(shuffled)
    n_train_pert = int(round(len(shuffled) * args.holdout_train_pert_ratio))
    train_pert_set = set(shuffled[:n_train_pert].tolist())
    test_pert_set = set(shuffled[n_train_pert:].tolist())

    # Split rule:
    # - train: all non-holdout cell lines + holdout cells whose perturbation in selected 30%
    # - test: holdout cells with perturbation in remaining 70% + all holdout controls
    in_holdout_train_pert = np.array([p in train_pert_set for p in pert], dtype=bool)
    in_holdout_test_pert = np.array([p in test_pert_set for p in pert], dtype=bool)
    holdout_control = holdout_mask & (pert == args.control_label)

    train_mask = non_holdout_mask | (holdout_mask & in_holdout_train_pert)
    test_mask = (holdout_mask & in_holdout_test_pert) | holdout_control

    train_adata = adata[train_mask].copy()
    test_adata = adata[test_mask].copy()

    all_path = args.output_dir / "replogle_hvg2000_div10.h5ad"
    train_path = args.output_dir / "replogle_hvg2000_div10_train.h5ad"
    test_path = args.output_dir / "replogle_hvg2000_div10_test.h5ad"
    summary_path = args.output_dir / "replogle_hvg2000_div10_split_summary.json"

    adata.write_h5ad(all_path)
    train_adata.write_h5ad(train_path)
    test_adata.write_h5ad(test_path)

    summary = {
        "input_h5ad": str(args.input_h5ad),
        "all_shape": [int(adata.n_obs), int(adata.n_vars)],
        "train_shape": [int(train_adata.n_obs), int(train_adata.n_vars)],
        "test_shape": [int(test_adata.n_obs), int(test_adata.n_vars)],
        "holdout_cell_line": args.holdout_cell_line,
        "control_label": args.control_label,
        "holdout_train_pert_ratio": args.holdout_train_pert_ratio,
        "n_holdout_perturbations_total": int(len(holdout_perturbations)),
        "n_holdout_perturbations_train": int(len(train_pert_set)),
        "n_holdout_perturbations_test": int(len(test_pert_set)),
        "train_perturbations_from_holdout": sorted(train_pert_set),
        "test_perturbations_from_holdout": sorted(test_pert_set),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved: {all_path}")
    print(f"saved: {train_path}")
    print(f"saved: {test_path}")
    print(f"saved: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

