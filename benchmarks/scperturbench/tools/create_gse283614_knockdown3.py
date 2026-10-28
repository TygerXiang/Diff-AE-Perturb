"""Create 3 GSE283614 folds whose test set is a *knockdown stratum*, not a spread.

Why this exists
---------------
The original ``dose_cv`` folds are rank-stratified *the other way round*: guides
are dealt round-robin by knockdown rank, so every fold spans the whole range
(0.00-0.83).  That answers "can the model interpolate to an unseen guide of a
familiar strength".  This builder answers the stricter question the reviewer
asked for: **can it extrapolate to a strength band it never trained on?**

Design
------
* **Contiguous knockdown blocks, not round-robin.**  Guides are sorted by
  knockdown fraction *within each target* and cut into three contiguous rank
  blocks: low / mid / high.  Fold 0 holds out the low block, fold 1 the mid block,
  fold 2 the high block.  A test set is therefore one strength band, and the
  training set is the other two bands - the model never sees that band.
* **Stratified by target.**  NANOG and OCT4 have wildly different knockdown
  distributions (OCT4 has three guides at exactly 0 and most below 0.2; NANOG is
  almost all above 0.3).  Splitting the pooled list would confound target with
  strength, so each target contributes its own low/mid/high blocks.  With 17 NANOG
  and 16 OCT4 guides that is 5-6 guides per block, i.e. 11-12 guides per fold.
* **Validation is the next block up, rotated.**  ``val_fold = (fold + 1) % 3``, as
  in the original builder, so validation is also an unseen band and never
  overlaps training or the test block.  Model selection is not used for the
  headline numbers anyway (the final-epoch checkpoint is evaluated), so the
  rotation only matters for ``checkpoint_best.pt``.
* **Controls are a separate, disjoint random 60/20/20 cell split**, exactly as in
  the original folds.  Contiguous blocks give unequal block sizes, so this axis
  cannot be stratified.

Consequence that must be reported
---------------------------------
Because a fold is defined by strength, its test set is *not* representative of
the guide population: fold 2's test guides are the strongest knockdns in the
dataset.  A per-fold effect size is therefore not an unbiased estimate of the
pooled one, and the "extrapolate to an unseen band" condition is strictly harder
than the original "interpolate to an unseen guide of familiar strength"
condition.  Pooling the three folds still covers all 33 guides exactly once, and
that pooled comparison is the pre-registered headline.

Writes a self-describing manifest plus, unless ``--manifest-only``, the three
per-fold h5ad triplets under ``<output-dir>/<state>/fold_<k>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path("/root/Diff-AE-Perturb")
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

N_FOLDS = 3
STRATUM_NAMES = {0: "low", 1: "mid", 2: "high"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build knockdown-stratum 3-fold GSE283614 splits.")
    parser.add_argument("--adata-path", type=Path, required=True)
    parser.add_argument("--dosage-table", type=Path, required=True,
                        help="gse283614_tf_dosage.tsv from the original dose_cv build.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def band_assignment(dosage: pd.DataFrame, n_folds: int) -> dict[str, int]:
    """Contiguous knockdown blocks within each target.

    Guides are sorted by knockdown fraction inside their target and cut into
    ``n_folds`` **contiguous rank blocks**: all of block 0 becomes fold 0, and so
    on.  Contiguity is the whole point - dealing the ranks round-robin (0,1,2,0,...)
    interleaves three rank blocks and leaves every fold spanning almost the full
    range again, which is the design this module exists to replace.

    The blocks are cut by rank *within* the guides present in each target, so the
    NANOG and OCT4 cut points differ; that is intended, because their knockdown
    distributions differ by an order of magnitude.
    """

    assignment: dict[str, int] = {}
    for target, group in dosage.groupby("dose_target", sort=True):
        ordered = group.sort_values(["knockdown_fraction", "guide_identity"]).reset_index(drop=True)
        n = len(ordered)
        for rank, row in ordered.iterrows():
            # Integer block index; the final block absorbs the remainder so no
            # guide is dropped when n is not divisible by n_folds.
            fold = min(int(rank * n_folds // n), n_folds - 1)
            assignment[str(row["guide_identity"])] = fold
    return assignment


def control_partition(indices: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    """Identical to the original builder: a disjoint random 60/20/20 cell split."""

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


def main() -> int:
    args = parse_args()
    import anndata as ad

    adata = ad.read_h5ad(args.adata_path)
    for column in ("cell_state", "guide_identity", "perturbation_label"):
        if column not in adata.obs:
            raise KeyError(f"Column {column!r} not found in adata.obs")

    dosage = pd.read_csv(args.dosage_table, sep="\t")
    dosage = dosage[["guide_identity", "dose_target", "knockdown_fraction"]].copy()
    if dosage["guide_identity"].duplicated().any():
        raise ValueError("dosage table has duplicate guide_identity rows")
    assignment = band_assignment(dosage, args.n_folds)
    dose_lookup = dict(zip(dosage["guide_identity"].astype(str), dosage["knockdown_fraction"]))
    target_lookup = dict(zip(dosage["guide_identity"].astype(str), dosage["dose_target"]))

    guides_all = adata.obs["guide_identity"].astype(str).to_numpy()
    labels_all = adata.obs["perturbation_label"].astype(str).to_numpy()
    states_all = adata.obs["cell_state"].astype(str).to_numpy()

    manifest: dict[str, object] = {
        "design": "knockdown_stratum_3fold",
        "adata_path": str(args.adata_path),
        "dosage_table": str(args.dosage_table),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "materialized": not args.manifest_only,
        "stratum_names": {str(k): v for k, v in STRATUM_NAMES.items()},
        "fold_assignment": assignment,
        "notes": [
            "Each fold holds out one contiguous knockdown block per target (low / mid / high).",
            "Training is the other two blocks, so the held-out band is never trained on.",
            "Validation is the next block up (val_fold = (fold + 1) % 3), also an unseen band.",
            "Controls use a disjoint random 60/20/20 cell split, as in the original folds.",
            "A per-fold effect size is NOT an unbiased estimate of the pooled one: the test block is not a random sample of guides.",
            "This is an extrapolation-to-unseen-strength design; the original dose_cv folds were interpolation-to-unseen-guide.",
        ],
        "states": {},
    }

    for state_idx, state in enumerate(sorted(np.unique(states_all))):
        state_positions = np.flatnonzero(states_all == state)
        state_guides = guides_all[state_positions]
        state_labels = labels_all[state_positions]
        controls = control_partition(np.flatnonzero(state_labels == "control"), seed=args.seed + state_idx)

        targeting_in_state = set(state_guides[state_labels != "control"])
        missing = sorted(set(assignment) - targeting_in_state)
        if missing:
            raise ValueError(f"State {state!r} is missing targeting guides: {missing}")

        state_manifest: dict[str, object] = {
            "n_cells": int(len(state_positions)),
            "n_controls": int((state_labels == "control").sum()),
            "folds": {},
        }
        state_dir = args.output_dir / state.lower()

        for fold in range(args.n_folds):
            val_fold = (fold + 1) % args.n_folds
            test_guides = {g for g, f in assignment.items() if f == fold}
            val_guides = {g for g, f in assignment.items() if f == val_fold}
            train_guides = set(assignment) - test_guides - val_guides
            overlap = (train_guides & test_guides) | (train_guides & val_guides) | (test_guides & val_guides)
            if overlap:
                raise ValueError(f"fold {fold} guide partitions overlap: {sorted(overlap)}")
            if not test_guides:
                raise ValueError(f"fold {fold} has an empty test set")

            masks = {
                "train": np.asarray([g in train_guides for g in state_guides]),
                "val": np.asarray([g in val_guides for g in state_guides]),
                "test": np.asarray([g in test_guides for g in state_guides]),
            }
            indices: dict[str, np.ndarray] = {}
            for split_name in ("train", "val", "test"):
                target_local = np.flatnonzero(masks[split_name] & (state_labels != "control"))
                local = np.sort(np.concatenate([target_local, controls[split_name]]))
                indices[split_name] = state_positions[local]

            fold_dir = state_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            paths: dict[str, str] = {}
            for split_name, positions in indices.items():
                path = fold_dir / f"gse283614_{state.lower()}_{split_name}.h5ad"
                if not args.manifest_only:
                    adata[positions].copy().write_h5ad(path)
                paths[split_name] = str(path)

            def describe(guide_set: set[str]) -> dict[str, object]:
                per_target: dict[str, object] = {}
                for target in sorted({target_lookup[g] for g in guide_set}):
                    values = sorted(dose_lookup[g] for g in guide_set if target_lookup[g] == target)
                    per_target[target] = {
                        "n_guides": len(values),
                        "min": values[0] if values else None,
                        "max": values[-1] if values else None,
                        "values": values,
                    }
                all_values = sorted(dose_lookup[g] for g in guide_set)
                return {
                    "n_guides": len(guide_set),
                    "overall_min": all_values[0] if all_values else None,
                    "overall_max": all_values[-1] if all_values else None,
                    "per_target": per_target,
                }

            state_manifest["folds"][str(fold)] = {
                "stratum": STRATUM_NAMES[fold],
                "validation_fold": val_fold,
                "paths": paths,
                "guides": {
                    "train": sorted(train_guides),
                    "val": sorted(val_guides),
                    "test": sorted(test_guides),
                },
                "knockdown": {
                    "train": describe(train_guides),
                    "val": describe(val_guides),
                    "test": describe(test_guides),
                },
                "cell_counts": {k: int(len(v)) for k, v in indices.items()},
                "control_cell_counts": {k: int(len(v)) for k, v in controls.items()},
                "guide_leakage": False,
            }
        manifest["states"][state] = state_manifest

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "knockdown3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== fold composition (knockdown bands must not overlap within a target) ===")
    for state, sm in manifest["states"].items():
        print(f"-- {state}")
        for fold, fm in sm["folds"].items():
            for split in ("train", "val", "test"):
                d = fm["knockdown"][split]
                parts = [
                    f"{t}[{v['min']:.3f},{v['max']:.3f}]n={v['n_guides']}"
                    for t, v in sorted(d["per_target"].items())
                ]
                tag = fm["stratum"] if split == "test" else ""
                print(f"   fold{fold} {split:5s} {tag:4s} n={d['n_guides']:2d}  " + "  ".join(parts))
        print(f"   cells: " + " | ".join(
            f"fold{f}: train={fm['cell_counts']['train']} test={fm['cell_counts']['test']}"
            for f, fm in sm["folds"].items()
        ))
    print()
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
