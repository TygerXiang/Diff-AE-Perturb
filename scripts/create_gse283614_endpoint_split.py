from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.gse283614_biology import (
    build_endpoint_interpolation_partition,
    read_author_dosage_table,
    validate_dosage_coverage,
    validate_fold_partition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create strict endpoint-trained, middle-dose-test GSE283614 splits."
    )
    parser.add_argument("--adata-path", type=Path, required=True)
    parser.add_argument("--dosage-workbook", type=Path, required=True)
    parser.add_argument("--guide-library-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def partition_controls(indices: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    n_train = int(round(0.60 * len(shuffled)))
    n_val = int(round(0.20 * len(shuffled)))
    return {
        "train": np.sort(shuffled[:n_train]),
        "val": np.sort(shuffled[n_train : n_train + n_val]),
        "test": np.sort(shuffled[n_train + n_val :]),
    }


def guide_rows(guides: set[str], dosage) -> list[dict[str, object]]:
    columns = [
        "guide_identity",
        "dose_target",
        "tf_expression_pct_control",
        "tf_remaining",
        "knockdown_fraction",
    ]
    return (
        dosage.loc[dosage["guide_identity"].isin(guides), columns]
        .sort_values(["dose_target", "knockdown_fraction", "guide_identity"])
        .to_dict("records")
    )


def main() -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("This script requires anndata") from exc

    args = parse_args()
    adata = ad.read_h5ad(args.adata_path)
    for column in ("cell_state", "guide_identity", "perturbation_label"):
        if column not in adata.obs:
            raise KeyError(f"Column {column!r} not found in adata.obs")

    dosage = read_author_dosage_table(
        args.dosage_workbook,
        guide_library_workbook=args.guide_library_workbook,
    )
    coverage = validate_dosage_coverage(dosage, adata.obs["guide_identity"].astype(str).unique())
    partitions = build_endpoint_interpolation_partition(
        dosage, endpoint_fraction=args.endpoint_fraction
    )
    validate_fold_partition(partitions["train"], partitions["val"], partitions["test"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targeting = dosage.loc[~dosage["is_control"]].copy()
    assignment = targeting[[
        "guide_identity",
        "dose_target",
        "tf_expression_pct_control",
        "tf_remaining",
        "knockdown_fraction",
    ]].copy()
    split_lookup = {
        guide: split for split, guides in partitions.items() for guide in guides
    }
    assignment["split"] = assignment["guide_identity"].map(split_lookup)
    assignment.to_csv(args.output_dir / "guide_split_assignments.csv", index=False)
    assignment.to_csv(args.output_dir / "gse283614_tf_dosage.tsv", sep="\t", index=False)

    states = adata.obs["cell_state"].astype(str).to_numpy()
    labels = adata.obs["perturbation_label"].astype(str).to_numpy()
    guides = adata.obs["guide_identity"].astype(str).to_numpy()
    manifest: dict[str, object] = {
        "design": "endpoint_train_middle_test",
        "adata_path": str(args.adata_path),
        "endpoint_fraction": args.endpoint_fraction,
        "seed": args.seed,
        "materialized": not args.manifest_only,
        "coverage": coverage,
        "guide_partitions": {
            split: guide_rows(values, targeting) for split, values in partitions.items()
        },
        "states": {},
    }

    for state_index, state in enumerate(sorted(np.unique(states))):
        state_positions = np.flatnonzero(states == state)
        state_labels = labels[state_positions]
        state_guides = guides[state_positions]
        observed_targeting = set(state_guides[state_labels != "control"])
        missing = sorted(set(split_lookup) - observed_targeting)
        if missing:
            raise ValueError(f"State {state!r} is missing targeting guides: {missing}")

        control_local = np.flatnonzero(state_labels == "control")
        controls = partition_controls(control_local, seed=args.seed + state_index)
        state_dir = args.output_dir / state.lower()
        state_dir.mkdir(parents=True, exist_ok=True)
        state_summary: dict[str, object] = {"n_cells": len(state_positions), "splits": {}}
        for split in ("train", "val", "test"):
            target_local = np.flatnonzero(
                np.asarray([guide in partitions[split] for guide in state_guides])
                & (state_labels != "control")
            )
            local = np.sort(np.concatenate([target_local, controls[split]]))
            positions = state_positions[local]
            path = state_dir / f"gse283614_{state.lower()}_{split}.h5ad"
            if not args.manifest_only:
                adata[positions].copy().write_h5ad(path)
            state_summary["splits"][split] = {
                "path": str(path),
                "n_cells": int(len(positions)),
                "n_targeting_cells": int(len(target_local)),
                "n_control_cells": int(len(controls[split])),
                "n_guides": len(partitions[split]),
            }
        manifest["states"][state] = state_summary

    manifest_path = args.output_dir / "endpoint_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
