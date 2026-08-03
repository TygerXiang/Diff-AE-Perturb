from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.gse283614_biology import (
    build_rank_stratified_guide_folds,
    read_author_dosage_table,
    validate_dosage_coverage,
    validate_fold_partition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-free state-specific GSE283614 dose CV folds.")
    parser.add_argument("--adata-path", type=Path, required=True)
    parser.add_argument("--dosage-workbook", type=Path, required=True)
    parser.add_argument("--guide-library-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Optional subset of outer folds to materialize.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manifest-only", action="store_true", help="Validate folds without writing h5ad files.")
    return parser.parse_args()


def control_partition(indices: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    n = len(shuffled)
    train_end = int(round(0.60 * n))
    val_end = train_end + int(round(0.20 * n))
    return {
        "train": np.sort(shuffled[:train_end]),
        "val": np.sort(shuffled[train_end:val_end]),
        "test": np.sort(shuffled[val_end:]),
    }


def dose_range(guide_set: set[str], dose_lookup: dict[str, float]) -> dict[str, float | int | None]:
    values = [dose_lookup[guide] for guide in guide_set if guide in dose_lookup]
    return {
        "n_guides": len(guide_set),
        "min_knockdown": None if not values else float(min(values)),
        "max_knockdown": None if not values else float(max(values)),
    }


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
    assignments = build_rank_stratified_guide_folds(dosage, n_folds=args.n_folds)
    targeting_dose = dosage.loc[~dosage["is_control"], [
        "guide_identity",
        "dose_target",
        "tf_expression_pct_control",
        "tf_remaining",
        "knockdown_fraction",
    ]].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targeting_dose.to_csv(args.output_dir / "gse283614_tf_dosage.tsv", sep="\t", index=False)
    assignment_rows = targeting_dose.copy()
    assignment_rows["outer_fold"] = assignment_rows["guide_identity"].map(assignments)
    assignment_rows.to_csv(args.output_dir / "guide_fold_assignments.csv", index=False)

    guides_all = adata.obs["guide_identity"].astype(str).to_numpy()
    labels_all = adata.obs["perturbation_label"].astype(str).to_numpy()
    states_all = adata.obs["cell_state"].astype(str).to_numpy()
    dose_lookup = dict(zip(targeting_dose["guide_identity"], targeting_dose["knockdown_fraction"]))
    target_lookup = dict(zip(targeting_dose["guide_identity"], targeting_dose["dose_target"]))
    manifest: dict[str, object] = {
        "adata_path": str(args.adata_path),
        "n_folds": args.n_folds,
        "selected_folds": list(range(args.n_folds)) if args.folds is None else args.folds,
        "seed": args.seed,
        "coverage": coverage,
        "materialized": not args.manifest_only,
        "fold_assignment": assignments,
        "states": {},
    }

    for state_idx, state in enumerate(sorted(np.unique(states_all))):
        state_positions = np.flatnonzero(states_all == state)
        state_guides = guides_all[state_positions]
        state_labels = labels_all[state_positions]
        control_local = np.flatnonzero(state_labels == "control")
        controls = control_partition(control_local, seed=args.seed + state_idx)
        state_manifest: dict[str, object] = {
            "n_cells": len(state_positions),
            "n_controls": len(control_local),
            "folds": {},
        }
        state_dir = args.output_dir / state.lower()
        state_dir.mkdir(parents=True, exist_ok=True)
        targeting_guides_in_state = set(state_guides[state_labels != "control"])
        missing_targeting = sorted(set(assignments) - targeting_guides_in_state)
        if missing_targeting:
            raise ValueError(f"State {state!r} is missing targeting guides: {missing_targeting}")

        selected_folds = range(args.n_folds) if args.folds is None else args.folds
        for outer_fold in selected_folds:
            if not 0 <= outer_fold < args.n_folds:
                raise ValueError(f"Fold {outer_fold} is outside [0, {args.n_folds})")
            val_fold = (outer_fold + 1) % args.n_folds
            test_guides = {guide for guide, fold in assignments.items() if fold == outer_fold}
            val_guides = {guide for guide, fold in assignments.items() if fold == val_fold}
            train_guides = set(assignments) - test_guides - val_guides
            validate_fold_partition(train_guides, val_guides, test_guides)
            target_masks = {
                "train": np.asarray([guide in train_guides for guide in state_guides]),
                "val": np.asarray([guide in val_guides for guide in state_guides]),
                "test": np.asarray([guide in test_guides for guide in state_guides]),
            }
            split_indices: dict[str, np.ndarray] = {}
            for split_name in ("train", "val", "test"):
                target_local = np.flatnonzero(target_masks[split_name] & (state_labels != "control"))
                local = np.sort(np.concatenate([target_local, controls[split_name]]))
                split_indices[split_name] = state_positions[local]

            fold_dir = state_dir / f"fold_{outer_fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            paths: dict[str, str] = {}
            for split_name, positions in split_indices.items():
                path = fold_dir / f"gse283614_{state.lower()}_{split_name}.h5ad"
                if not args.manifest_only:
                    adata[positions].copy().write_h5ad(path)
                paths[split_name] = str(path)

            target_counts = {
                split: {
                    target: sum(target_lookup[guide] == target for guide in guides)
                    for target in ("NANOG", "OCT4")
                }
                for split, guides in (("train", train_guides), ("val", val_guides), ("test", test_guides))
            }
            state_manifest["folds"][str(outer_fold)] = {
                "validation_fold": val_fold,
                "paths": paths,
                "guides": {
                    "train": sorted(train_guides),
                    "val": sorted(val_guides),
                    "test": sorted(test_guides),
                },
                "target_counts": target_counts,
                "dose_ranges": {
                    "train": dose_range(train_guides, dose_lookup),
                    "val": dose_range(val_guides, dose_lookup),
                    "test": dose_range(test_guides, dose_lookup),
                },
                "cell_counts": {split: int(len(values)) for split, values in split_indices.items()},
                "control_cell_counts": {split: int(len(values)) for split, values in controls.items()},
                "guide_leakage": False,
            }
        manifest["states"][state] = state_manifest

    manifest_path = args.output_dir / "dose_cv_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
