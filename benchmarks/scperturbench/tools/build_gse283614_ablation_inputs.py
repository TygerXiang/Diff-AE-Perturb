"""Build the GSE283614 activity-ablation input artifacts.

This is the GSE283614 counterpart of the GSE132080 ablation inputs.  GSE283614
has no Table_S8-style 26-mer CNN features for its enhancer/promoter tiling
guides, so the activity axis is the *measured* reference-screen depletion
(``Log2(Day 9 / Day 0)``), normalised to the strongest guide in the same region
(see ``Diff-AE-Perturb/agent.md`` section "GSE283614 Activity Mapping").

Outputs, all under ``--output-dir``:

* ``gse283614_activity_<state>.tsv``  label + state-specific relative activity.
  The first column is literally named ``guide_identity`` because both
  ``perturbdiff.data.load_guide_activity_lookup`` and
  ``perturbdiff.guide_activity.load_guide_score_lookup`` default to the first
  column as the guide label.

  **Unmeasured guides are written as an explicit 0.0, and that is a deliberate
  choice with a cost.**  Writing them as an empty field instead makes
  ``load_guide_activity_lookup`` skip the row, after which
  ``attach_direct_activity_features`` raises on the first non-control label it
  cannot resolve - which is how this builder originally behaved and which the
  preflight now rejects, because it would make the published 2026-07-28
  GSE283614 comparison unreproducible ("missing activity is passed as 0",
  ``Diff-AE-Perturb/agent.md``).  Filling with 0 keeps both arms trainable but
  means an unmeasured guide is numerically indistinguishable from a guide whose
  screen depletion really was zero.  ``conditions_<state>.json`` records
  ``activity_measured`` per guide so the report can split those two groups, and
  the preflight verifies that the split is still recoverable from the table.
* ``conditions_<state>.json``  per-fold evaluated conditions, test-cell counts
  and per-guide activity provenance, in the same schema as
  ``inputs/gse132080_bench/conditions.json``.
* ``genes.json``  the shared gene order of every fold file.
* ``deg_table_<state>_fold<k>.json``  ``myUtil1.calDEG`` ground-truth DEG table
  for that fold's held-out guides, computed on train + test ground-truth cells
  only.

Everything is a pure read of the prepared dataset plus the fold manifest; no
file outside ``--output-dir`` is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_LABEL = "control"

# Column in gse283614_table_s1_activity.tsv that carries the state-specific
# relative activity.  ``relative_activity_mean`` deliberately is not used: it
# averages two screens whose reference effects differ (5.30 for NANOG e1, 7.47
# for OCT4 PE), which mixes incompatible normalisations.
STATE_ACTIVITY_COLUMN = {
    "ESC": "relative_activity_ESC",
    "DE": "relative_activity_DE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GSE283614 ablation inputs.")
    parser.add_argument("--cv-dir", required=True, type=Path,
                        help="Directory holding leakage-free guide folds, laid out as <state>/fold_<k>/.")
    parser.add_argument("--manifest", default="dose_cv_manifest.json",
                        help="Fold manifest inside --cv-dir (use knockdown3_manifest.json for the stratum folds).")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Fold indices to include. Defaults to every fold in the manifest.")
    parser.add_argument("--deg-prefix", default="deg_table",
                        help="Prefix for the per-fold DEG tables, so different fold sets do not collide.")
    parser.add_argument("--activity-table", required=True, type=Path,
                        help="gse283614_table_s1_activity.tsv.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--states", nargs="+", default=["ESC", "DE"])
    parser.add_argument("--min-test-cells", type=int, default=20)
    parser.add_argument("--skip-deg", action="store_true",
                        help="Only build the activity/conditions/genes artifacts.")
    parser.add_argument("--keep-activity-tables", action="store_true",
                        help="Do not fall back to an existing activity table when one is already present.")
    return parser.parse_args()


def load_activity(table_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(table_path, sep="\t")
    if "guide_identity" not in frame.columns:
        raise KeyError(f"{table_path} has no guide_identity column")
    return frame.set_index("guide_identity", drop=False)


def write_activity_table(table: pd.DataFrame, column: str, state: str, output_dir: Path) -> dict:
    """Write the loader-shaped activity TSV and report exact coverage.

    Unmeasured *targeting* guides are filled with 0.0 so that
    ``attach_direct_activity_features`` can resolve every non-control label; see
    the module docstring for why this is a deliberate trade-off.
    """

    labels = table["guide_identity"].astype(str)
    raw = pd.to_numeric(table[column], errors="coerce")
    targeting = table["target_gene"].astype(str) != CONTROL_LABEL
    # Controls carry no activity channel at all (the trainer forces 0 for id 0 and
    # the model zeroes control rows again in _build_activity_embedding), so their
    # screen values are dropped rather than forwarded.
    values = raw.where(targeting, 0.0)
    out = pd.DataFrame(
        {
            "guide_identity": labels,
            "relative_activity": values,
            "target_gene": table["target_gene"].astype(str),
            "guide_type": table["guide_type"].astype(str),
        }
    )
    path = output_dir / f"gse283614_activity_{state.lower()}.tsv"
    out.to_csv(path, sep="\t", index=False, na_rep="0.0")

    measured = raw.notna() & targeting
    t_measured = measured
    return {
        "path": str(path),
        "activity_column": column,
        "n_rows": int(len(out)),
        "n_targeting": int(targeting.sum()),
        "n_measured": int(t_measured.sum()),
        "n_missing": int((targeting & ~measured).sum()),
        "measured_guides": sorted(out.loc[t_measured, "guide_identity"].tolist()),
        "missing_guides": sorted(out.loc[targeting & ~measured, "guide_identity"].tolist()),
        "n_measured_positive": int((out.loc[t_measured, "relative_activity"] > 0).sum()),
        "n_measured_zero": int((out.loc[t_measured, "relative_activity"] == 0).sum()),
        "n_unmeasured_filled_zero": int((targeting & ~measured).sum()),
        "n_control_guides_forced_zero": int((~targeting).sum()),
        "missing_value_policy": "unmeasured targeting guides are written as 0.0 (reproducible with the published 2026-07-28 GSE283614 comparison); conditions_<state>.json keeps activity_measured so the split stays recoverable",
    }


def build_conditions(
    *,
    state: str,
    fold_summaries: list[dict],
    activity: pd.DataFrame,
    activity_column: str,
    min_test_cells: int,
) -> dict:
    per_condition: dict[str, dict] = {}
    evaluated: list[str] = []
    for fold in fold_summaries:
        for guide, detail in sorted(fold["per_condition"].items()):
            raw = activity.loc[guide, activity_column] if guide in activity.index else np.nan
            measured = bool(pd.notna(raw))
            per_condition[guide] = {
                "fold": int(fold["fold"]),
                "n_test_cells": int(detail["n_test_cells"]),
                "n_train_cells": int(detail["n_train_cells"]),
                # Every evaluated guide is held out of its own fold's training set.
                "seen_in_train": False,
                "target_gene": str(detail["target_gene"]),
                "guide_type": str(detail["guide_type"]),
                "activity_column": activity_column,
                # Recorded as the model will see it: an unmeasured guide becomes 0.
                "activity_input": 0.0 if not measured else float(raw),
                "activity_measured": measured,
                "activity_raw_or_none": None if not measured else float(raw),
                "evaluated": int(detail["n_test_cells"]) >= int(min_test_cells),
            }
            if int(detail["n_test_cells"]) >= int(min_test_cells):
                evaluated.append(guide)

    return {
        "dataset": "GSE283614",
        "state": state,
        "control_label": CONTROL_LABEL,
        "min_test_cells": int(min_test_cells),
        "activity_source": "reference-screen log2FC depletion (measured; not a prediction)",
        "activity_column": activity_column,
        "eval_level": "guide (held-out guide per fold, out-of-fold pooled)",
        "n_test_conditions_total": len(per_condition),
        "n_train_conditions": sum(len(f["train_guides"]) for f in fold_summaries),
        "evaluated_conditions": evaluated,
        "n_evaluated": len(evaluated),
        "n_evaluated_measured_activity": sum(
            1 for g in evaluated if per_condition[g]["activity_measured"]
        ),
        "excluded_conditions": {
            guide: int(detail["n_test_cells"])
            for guide, detail in per_condition.items()
            if not detail["evaluated"]
        },
        "n_test_control_cells": int(sum(f["n_test_control_cells"] for f in fold_summaries)),
        "n_train_control_cells": int(sum(f["n_train_control_cells"] for f in fold_summaries)),
        "folds": fold_summaries,
        "per_condition": per_condition,
    }


def build_deg_table(adatas: list, targets: list[str], control: str = CONTROL_LABEL) -> dict:
    """calDEG port: t-test vs control, foldchanges = 2**logfoldchanges, by |score|.

    Identical recipe to ``scperturbench_inputs.build_deg_table``; reimplemented
    here so this builder stays dependency-free apart from scanpy and so the
    GSE283614 payload can tag the right dataset name.
    """

    import anndata as ad
    import scanpy as sc

    full = adatas[0] if len(adatas) == 1 else ad.concat(adatas, join="inner")
    table: dict[str, dict] = {}
    for target in targets:
        sub = full[full.obs["perturbation_label"].astype(str).isin([target, control])].copy()
        if sub.n_obs == 0:
            continue
        sc.tl.rank_genes_groups(
            sub, "perturbation_label", groups=[target], reference=control, method="t-test"
        )
        result = sub.uns["rank_genes_groups"]
        frame = pd.DataFrame(
            {
                "names": result["names"][target],
                "pvals_adj": result["pvals_adj"][target],
                "logfoldchanges": result["logfoldchanges"][target],
                "scores": result["scores"][target],
            }
        )
        frame["foldchanges"] = 2 ** frame["logfoldchanges"]
        frame = frame.drop(columns=["logfoldchanges"]).set_index("names")
        table[target] = {
            "order": frame.index.astype(str).tolist(),
            "pvals_adj": [None if pd.isna(v) else float(v) for v in frame["pvals_adj"]],
            "foldchanges": [None if pd.isna(v) else float(v) for v in frame["foldchanges"]],
            "scores": [None if pd.isna(v) else float(v) for v in frame["scores"]],
        }
    return table


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import anndata as ad

    manifest_path = args.cv_dir / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    activity = load_activity(args.activity_table)
    selected = set(args.folds) if args.folds else None

    coverage: dict[str, object] = {}
    genes_written = False

    for state in [s.upper() for s in args.states]:
        column = STATE_ACTIVITY_COLUMN[state]
        coverage[state] = write_activity_table(activity, column, state, args.output_dir)

        state_key = state
        if state_key not in manifest["states"]:
            raise KeyError(f"state {state!r} not present in {manifest_path}")
        folds_available = {
            int(k): v for k, v in manifest["states"][state_key]["folds"].items()
        }
        if selected is not None:
            unknown = sorted(selected - set(folds_available))
            if unknown:
                raise KeyError(f"folds {unknown} not in {manifest_path} (has {sorted(folds_available)})")
        fold_summaries: list[dict] = []
        for fold_idx in sorted(folds_available):
            if selected is not None and fold_idx not in selected:
                continue
            fold = folds_available[fold_idx]
            fold_key = str(fold_idx)
            if not fold.get("guide_leakage") is False:
                raise ValueError(f"fold {fold_key} of {state} is not leakage-free")
            fold_dir = args.cv_dir / state.lower() / f"fold_{fold_key}"
            prefix = f"gse283614_{state.lower()}"
            train_path = fold_dir / f"{prefix}_train.h5ad"
            test_path = fold_dir / f"{prefix}_test.h5ad"
            train = ad.read_h5ad(train_path, backed="r")
            test = ad.read_h5ad(test_path, backed="r")

            if not genes_written:
                (args.output_dir / "genes.json").write_text(
                    json.dumps(
                        {
                            "genes": [str(g) for g in test.var_names],
                            "source": str(test_path),
                            "n_genes": int(test.n_vars),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                genes_written = True

            train_guides = sorted(fold["guides"]["train"])
            test_guides = sorted(fold["guides"]["test"])
            overlap = sorted(set(train_guides) & set(test_guides))
            if overlap:
                raise ValueError(f"fold {fold_key} of {state} reuses held-out guides: {overlap}")

            test_labels = test.obs["perturbation_label"].astype(str)
            test_counts = test_labels.value_counts()
            per_condition = {}
            for guide in test_guides:
                n_test = int(test_counts.get(guide, 0))
                meta = test.obs.loc[
                    test_labels == guide, ["target_gene", "guide_type"]
                ].iloc[0]
                per_condition[guide] = {
                    "n_test_cells": n_test,
                    "n_train_cells": int(
                        (train.obs["perturbation_label"].astype(str) == guide).sum()
                    ),
                    "target_gene": str(meta["target_gene"]),
                    "guide_type": str(meta["guide_type"]),
                }
            fold_summaries.append(
                {
                    "fold": int(fold_key),
                    "train_guides": train_guides,
                    "test_guides": test_guides,
                    "n_test_control_cells": int(fold["control_cell_counts"]["test"]),
                    "n_train_control_cells": int(fold["control_cell_counts"]["train"]),
                    "train_path": str(train_path),
                    "test_path": str(test_path),
                    "train_sha256": sha256(train_path),
                    "test_sha256": sha256(test_path),
                    "per_condition": per_condition,
                }
            )

            if not args.skip_deg:
                train_full = ad.read_h5ad(train_path)
                test_full = ad.read_h5ad(test_path)
                deg = build_deg_table([train_full, test_full], test_guides)
                missing = sorted(set(test_guides) - set(deg))
                if missing:
                    raise ValueError(f"fold {fold_key} DEG table missing {missing}")
                payload = {
                    "recipe": "myUtil1.calDEG (scanpy t-test vs control, foldchanges=2**logfoldchanges, order by |scores|)",
                    "dataset": "GSE283614",
                    "state": state,
                    "fold": int(fold_key),
                    "genes": deg,
                }
                (args.output_dir / f"{args.deg_prefix}_{state.lower()}_fold{fold_key}.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )

        conditions = build_conditions(
            state=state,
            fold_summaries=fold_summaries,
            activity=activity,
            activity_column=column,
            min_test_cells=args.min_test_cells,
        )
        (args.output_dir / f"conditions_{state.lower()}.json").write_text(
            json.dumps(conditions, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = {
        "dataset": "GSE283614",
        "cv_manifest": str(manifest_path),
        "activity_table": str(args.activity_table),
        "activity_table_sha256": sha256(args.activity_table),
        "coverage": coverage,
        "output_dir": str(args.output_dir),
        "deg_built": not args.skip_deg,
    }
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
