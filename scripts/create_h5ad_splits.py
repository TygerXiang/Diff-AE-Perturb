from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from a prepared .h5ad with stratification on an obs column."
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stratify-col", default="guide_identity")
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--prefix", default="gse132080")
    return parser.parse_args()


def summarize_counts(values: np.ndarray) -> dict[str, int]:
    uniq, counts = np.unique(values.astype(str), return_counts=True)
    return {str(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}


def main() -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("This script requires anndata.") from exc

    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    adata = ad.read_h5ad(args.input_h5ad)
    if args.stratify_col not in adata.obs:
        raise KeyError(f"Column {args.stratify_col!r} not found in adata.obs")

    labels = adata.obs[args.stratify_col].astype(str).to_numpy()
    indices = np.arange(adata.n_obs)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_ratio,
        random_state=args.split_seed,
        stratify=labels,
    )
    train_val_labels = labels[train_val_idx]
    val_adjusted = args.val_ratio / (1.0 - args.test_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_adjusted,
        random_state=args.split_seed,
        stratify=train_val_labels,
    )

    train_adata = adata[train_idx].copy()
    val_adata = adata[val_idx].copy()
    test_adata = adata[test_idx].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.prefix}_train.h5ad"
    val_path = args.output_dir / f"{args.prefix}_val.h5ad"
    test_path = args.output_dir / f"{args.prefix}_test.h5ad"
    summary_path = args.output_dir / f"{args.prefix}_split_summary.json"

    train_adata.write_h5ad(train_path)
    val_adata.write_h5ad(val_path)
    test_adata.write_h5ad(test_path)

    summary = {
        "input_h5ad": str(args.input_h5ad),
        "stratify_col": args.stratify_col,
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "n_obs_total": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "splits": {
            "train": {
                "path": str(train_path),
                "n_obs": int(train_adata.n_obs),
                "n_unique_strata": int(train_adata.obs[args.stratify_col].astype(str).nunique()),
                "strata_counts": summarize_counts(train_adata.obs[args.stratify_col].astype(str).to_numpy()),
            },
            "val": {
                "path": str(val_path),
                "n_obs": int(val_adata.n_obs),
                "n_unique_strata": int(val_adata.obs[args.stratify_col].astype(str).nunique()),
                "strata_counts": summarize_counts(val_adata.obs[args.stratify_col].astype(str).to_numpy()),
            },
            "test": {
                "path": str(test_path),
                "n_obs": int(test_adata.n_obs),
                "n_unique_strata": int(test_adata.obs[args.stratify_col].astype(str).nunique()),
                "strata_counts": summarize_counts(test_adata.obs[args.stratify_col].astype(str).to_numpy()),
            },
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved: {train_path}")
    print(f"saved: {val_path}")
    print(f"saved: {test_path}")
    print(f"saved: {summary_path}")
    print(
        json.dumps(
            {
                "train_n_obs": int(train_adata.n_obs),
                "val_n_obs": int(val_adata.n_obs),
                "test_n_obs": int(test_adata.n_obs),
                "train_unique": int(train_adata.obs[args.stratify_col].astype(str).nunique()),
                "val_unique": int(val_adata.obs[args.stratify_col].astype(str).nunique()),
                "test_unique": int(test_adata.obs[args.stratify_col].astype(str).nunique()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
