from __future__ import annotations

import argparse
from collections import Counter
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
    parser.add_argument("--split-mode", choices=("cell", "group"), default="cell")
    parser.add_argument("--group-col", default="guide_identity")
    parser.add_argument(
        "--group-stratify-col",
        default=None,
        help="Optional obs column used to stratify unique groups, e.g. guide_family.",
    )
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--prefix", default="gse132080")
    return parser.parse_args()


def summarize_counts(values: np.ndarray) -> dict[str, int]:
    uniq, counts = np.unique(values.astype(str), return_counts=True)
    return {str(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}


def majority_label(values: np.ndarray) -> str:
    counts = Counter(str(value) for value in values)
    return counts.most_common(1)[0][0]


def stratified_split_or_fallback(
    items: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    stratify: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if stratify is None:
        return train_test_split(items, test_size=test_size, random_state=random_state)
    labels, counts = np.unique(stratify.astype(str), return_counts=True)
    if len(labels) < 2 or int(counts.min()) < 2:
        return train_test_split(items, test_size=test_size, random_state=random_state)
    try:
        return train_test_split(items, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError:
        return train_test_split(items, test_size=test_size, random_state=random_state)


def split_group_indices(
    group_values: np.ndarray,
    *,
    split_seed: int,
    val_ratio: float,
    test_ratio: float,
    group_strata: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    unique_groups = np.unique(group_values.astype(str))
    if unique_groups.size < 3:
        raise ValueError("Group split requires at least three unique groups")

    strata_by_group = None
    if group_strata is not None:
        strata_by_group = np.array(
            [
                majority_label(group_strata[group_values.astype(str) == group])
                for group in unique_groups
            ],
            dtype=str,
        )

    train_val_groups, test_groups = stratified_split_or_fallback(
        unique_groups,
        test_size=test_ratio,
        random_state=split_seed,
        stratify=strata_by_group,
    )
    train_val_strata = None
    if strata_by_group is not None:
        strata_map = dict(zip(unique_groups.tolist(), strata_by_group.tolist()))
        train_val_strata = np.array([strata_map[group] for group in train_val_groups], dtype=str)
    val_adjusted = val_ratio / (1.0 - test_ratio)
    train_groups, val_groups = stratified_split_or_fallback(
        train_val_groups,
        test_size=val_adjusted,
        random_state=split_seed,
        stratify=train_val_strata,
    )

    train_group_set = set(train_groups.tolist())
    val_group_set = set(val_groups.tolist())
    test_group_set = set(test_groups.tolist())
    train_idx = np.flatnonzero(np.array([group in train_group_set for group in group_values.astype(str)]))
    val_idx = np.flatnonzero(np.array([group in val_group_set for group in group_values.astype(str)]))
    test_idx = np.flatnonzero(np.array([group in test_group_set for group in group_values.astype(str)]))
    metadata = {
        "n_unique_groups_total": int(unique_groups.size),
        "groups": {
            "train": sorted(train_group_set),
            "val": sorted(val_group_set),
            "test": sorted(test_group_set),
        },
    }
    return train_idx, val_idx, test_idx, metadata


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
    group_metadata: dict[str, object] | None = None
    if args.split_mode == "cell":
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
    else:
        if args.group_col not in adata.obs:
            raise KeyError(f"Column {args.group_col!r} not found in adata.obs")
        group_values = adata.obs[args.group_col].astype(str).to_numpy()
        group_strata = None
        if args.group_stratify_col is not None:
            if args.group_stratify_col not in adata.obs:
                raise KeyError(f"Column {args.group_stratify_col!r} not found in adata.obs")
            group_strata = adata.obs[args.group_stratify_col].astype(str).to_numpy()
        train_idx, val_idx, test_idx, group_metadata = split_group_indices(
            group_values,
            split_seed=args.split_seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            group_strata=group_strata,
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
        "split_mode": args.split_mode,
        "group_col": args.group_col if args.split_mode == "group" else None,
        "group_stratify_col": args.group_stratify_col if args.split_mode == "group" else None,
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
    if group_metadata is not None:
        summary["group_split"] = group_metadata
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
