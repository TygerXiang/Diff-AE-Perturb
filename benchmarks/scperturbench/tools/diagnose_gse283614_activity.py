"""Diagnose the GSE283614 activity axis before any GPU time is spent on it.

Answers four questions with numbers rather than assumptions:

1. **Coverage** - how many of the 33 targeting guides carry a measured activity
   value in each state, which ones do not, and what the model will therefore see
   for them (0, because ``attach_direct_activity_features`` fills controls and
   the missing-value path in the trainer is only reachable for unmeasured
   labels, which the activity table deliberately leaves empty).
2. **Degeneracy** - is the axis informative at all?  A channel whose values are
   mostly 0 cannot move the denoiser much, and a null result would then be
   uninformative rather than negative.
3. **Fold balance** - per fold, how many held-out guides carry activity, so the
   paired test set is known before the run and the partial-coverage sensitivity
   split is pre-registered rather than chosen after seeing results.
4. **Does the axis track anything real?** - Spearman correlation between the
   measured activity and the *observed* single-cell response magnitude of the
   same guide (L2 distance from the guide's mean expression to the state control
   mean).  This is a direct check of whether "screen depletion" carries any
   information about the transcriptomic perturbation being predicted.  It is a
   diagnostic, not a result: it uses held-out ground truth and must never be fed
   to the model.

Read-only: writes one JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Kept in sync with the builder: the state-specific screen column that carries the
# measured activity.  Duplicated rather than imported so this diagnostic can run
# without the builder module on the path.
STATE_ACTIVITY_COLUMN = {
    "esc": "relative_activity_ESC",
    "de": "relative_activity_DE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the GSE283614 activity axis.")
    parser.add_argument("--inputs-dir", required=True, type=Path,
                        help="Directory produced by build_gse283614_ablation_inputs.py.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--states", nargs="+", default=["esc", "de"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-cells-per-guide", type=int, default=400,
                        help="Subsample cap for the observed-response diagnostic.")
    parser.add_argument("--source-activity-table", type=Path, default=None,
                        help="The pre-fill source table (gse283614_table_s1_activity.tsv). When given, "
                             "the measured/unmeasured split is cross-checked against it so a silent "
                             "regression in the missing->0.0 fill cannot go unnoticed.")
    return parser.parse_args()


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and a two-sided p-value, without a scipy dependency."""

    from scipy import stats

    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def describe(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "n_zero": int((arr == 0).sum()),
        "n_unique": int(len(np.unique(arr))),
    }


def cross_check_against_source(
    source_path: Path, state: str, measured: list[str], unmeasured: list[str]
) -> dict:
    """Verify the measured/unmeasured split against the pre-fill source table.

    The source table leaves unmeasured guides blank; the builder fills them with
    0.0.  If the two disagree, the conditions file's ``activity_measured`` flag is
    wrong and every coverage number downstream inherits the error.
    """

    source = pd.read_csv(source_path, sep="\t")
    column = STATE_ACTIVITY_COLUMN[state]
    if column not in source.columns:
        raise KeyError(f"{source_path} has no {column!r} column")
    source = source.set_index("guide_identity")
    source_measured = {str(g) for g in source.index[source[column].notna()]}
    reported = set(measured)
    # Controls legitimately carry a screen value but are never conditioned on.
    targeting = set(measured) | set(unmeasured)
    false_positive = sorted(targeting & reported - source_measured)
    false_negative = sorted(targeting & source_measured - reported)
    return {
        "source_table": str(source_path),
        "source_column": column,
        "n_source_measured": len(source_measured),
        "false_positive_guides": false_positive,
        "false_negative_guides": false_negative,
        "agrees": not false_positive and not false_negative,
    }


def observed_response_magnitudes(
    test_paths: list[Path], guides: list[str], control: str, seed: int, cap: int
) -> dict[str, float]:
    """L2 distance from each guide's mean expression to the state control mean."""

    import anndata as ad

    rng = np.random.default_rng(seed)
    control_rows: list[np.ndarray] = []
    guide_rows: dict[str, list[np.ndarray]] = {g: [] for g in guides}
    for path in test_paths:
        adata = ad.read_h5ad(path)
        labels = adata.obs["perturbation_label"].astype(str).to_numpy()
        matrix = adata.X
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        dense = np.asarray(dense, dtype=np.float64)
        ctrl_idx = np.flatnonzero(labels == control)
        if ctrl_idx.size:
            take = rng.choice(ctrl_idx, min(cap, ctrl_idx.size), replace=False)
            control_rows.append(dense[np.sort(take)])
        for guide in guides:
            idx = np.flatnonzero(labels == guide)
            if idx.size == 0:
                continue
            take = rng.choice(idx, min(cap, idx.size), replace=False)
            guide_rows[guide].append(dense[np.sort(take)])
    if not control_rows:
        return {}
    control_mean = np.vstack(control_rows).mean(axis=0)
    out: dict[str, float] = {}
    for guide, blocks in guide_rows.items():
        if not blocks:
            continue
        mean = np.vstack(blocks).mean(axis=0)
        out[guide] = float(np.linalg.norm(mean - control_mean))
    return out


def activity_lookup(activity: pd.DataFrame) -> dict[str, float | None]:
    """Relative activity per guide, with ``None`` for an unmeasured guide."""

    out: dict[str, float | None] = {}
    for guide, value in activity["relative_activity"].items():
        out[str(guide)] = None if pd.isna(value) else float(value)
    return out


def measured_guides(conditions: dict) -> tuple[list[str], list[str]]:
    """Split evaluated guides into measured / unmeasured.

    The authoritative flag is ``activity_measured`` in the conditions file, which
    the builder sets from the *source* screen table before the missing->0.0 fill.
    Inferring it from the built activity table no longer works: unmeasured guides
    are written as 0.0 (see the builder docstring), so an empty-field test reports
    every guide as measured.  That regression is why this helper exists.
    """

    evaluated = list(conditions["evaluated_conditions"])
    per_condition = conditions["per_condition"]
    measured = [g for g in evaluated if per_condition[g]["activity_measured"]]
    unmeasured = [g for g in evaluated if not per_condition[g]["activity_measured"]]
    return measured, unmeasured


def main() -> int:
    args = parse_args()
    inputs = args.inputs_dir
    report: dict[str, object] = {
        "inputs_dir": str(inputs),
        "seed": args.seed,
        "states": {},
    }

    for state in [s.lower() for s in args.states]:
        conditions = json.loads((inputs / f"conditions_{state}.json").read_text(encoding="utf-8"))
        activity = pd.read_csv(
            inputs / f"gse283614_activity_{state}.tsv", sep="\t", keep_default_na=False,
            na_values=[""],
        ).set_index("guide_identity")
        lookup = activity_lookup(activity)
        evaluated = list(conditions["evaluated_conditions"])
        excluded = conditions["excluded_conditions"]

        measured, unmeasured = measured_guides(conditions)
        measured_set = set(measured)
        values = [float(activity.loc[g, "relative_activity"]) for g in measured]

        per_fold: dict[str, dict] = {}
        for fold in conditions["folds"]:
            fkey = f"fold{fold['fold']}"
            test_guides = [g for g in fold["test_guides"] if g in evaluated]
            per_fold[fkey] = {
                "n_test_guides": len(test_guides),
                "n_measured": sum(1 for g in test_guides if g in measured_set),
                "n_unmeasured": sum(1 for g in test_guides if g not in measured_set),
                "measured_guides": sorted(g for g in test_guides if g in measured_set),
                "unmeasured_guides": sorted(g for g in test_guides if g not in measured_set),
                "test_cell_counts": {
                    g: int(conditions["per_condition"][g]["n_test_cells"]) for g in test_guides
                },
                "guide_type_counts": {
                    "promoter": sum(
                        1 for g in test_guides if conditions["per_condition"][g]["guide_type"] == "promoter"
                    ),
                    "enhancer": sum(
                        1 for g in test_guides if conditions["per_condition"][g]["guide_type"] == "enhancer"
                    ),
                },
            }

        guide_types = {g: conditions["per_condition"][g]["guide_type"] for g in evaluated}
        unmeasured_types = sorted({guide_types[g] for g in unmeasured})

        # Observed-response diagnostic on the held-out ground truth.
        test_paths = [Path(f["test_path"]) for f in conditions["folds"]]
        magnitudes = observed_response_magnitudes(
            test_paths, evaluated, conditions["control_label"], args.seed, args.max_cells_per_guide
        )
        diag_rows = [
            {"guide": g, "activity": float(lookup[g]), "observed_response": magnitudes[g]}
            for g in measured
            if g in magnitudes
        ]
        rho_all, p_all = spearman(
            np.array([r["activity"] for r in diag_rows]),
            np.array([r["observed_response"] for r in diag_rows]),
        )
        # Within-target is the informative comparison: between-target spread is
        # driven by the library design (enhancer vs promoter), not by activity.
        # NANOG keeps its own symbol; POU5F1 is OCT4.
        target_of = {"NANOG": "NANOG", "POU5F1": "OCT4"}
        within: dict[str, dict] = {}
        for target in ("NANOG", "OCT4"):
            rows = [
                r for r in diag_rows
                if target_of.get(conditions["per_condition"][r["guide"]]["target_gene"]) == target
            ]
            rho, p = spearman(
                np.array([r["activity"] for r in rows]),
                np.array([r["observed_response"] for r in rows]),
            )
            within[target] = {"n": len(rows), "spearman_rho": rho, "p_value": p}

        report["states"][state] = {
            "activity_column": conditions["activity_column"],
            "n_evaluated": len(evaluated),
            "n_measured": len(measured),
            "n_unmeasured": len(unmeasured),
            "measured_source": "conditions_<state>.json::activity_measured (set from the pre-fill source table)",
            "unmeasured_guides": sorted(unmeasured),
            "unmeasured_guide_types": unmeasured_types,
            "excluded_by_min_test_cells": excluded,
            "activity_distribution": describe(values),
            "per_fold": per_fold,
            "source_cross_check": (
                cross_check_against_source(
                    args.source_activity_table, state, measured, unmeasured
                )
                if args.source_activity_table is not None
                else None
            ),
            "observed_response_diagnostic": {
                "definition": "L2 distance between guide mean expression and state control mean, held-out ground truth",
                "spearman_activity_vs_observed_response": rho_all,
                "p_value": p_all,
                "n": len(diag_rows),
                "within_target": within,
                "rows": diag_rows,
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
