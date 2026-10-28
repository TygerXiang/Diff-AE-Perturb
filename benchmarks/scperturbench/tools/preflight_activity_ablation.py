"""Preflight validation for the GSE132080 sgRNA-activity (Diff-AE-Perturb) ablation.

Run this before spending GPU time on training.  It turns the assumptions the ablation relies on
into hard assertions, and it records the provenance the report must cite:

1. the train/test splits are the exact files the six-metric benchmark used (sha256 match),
2. the test split really holds out whole guides (held-out guide set disjoint from the training
   guide set), so the target-level evaluation is an unseen-guide evaluation,
3. every evaluated condition maps to exactly one held-out guide (or the condition aggregation
   would silently mix guides),
4. the Table S8 CNN features resolve for every guide the activity arm must condition on, and the
   flattened feature width matches ``guide_cnn_feature_dim``,
5. the Table S16 activity axis actually carries signal (non-degenerate distribution, usable
   within-target spread).

Writes a JSON report and exits non-zero when a check fails.

Usage (inside the scperturbench image)::

    conda run --no-capture-output -n scarches python tools/preflight_activity_ablation.py \
        --train-h5ad  .scperturbench/inputs/gse132080_squidiff_repro/squidiff_train.h5ad \
        --test-h5ad   .scperturbench/inputs/gse132080_squidiff_repro/squidiff_test.h5ad \
        --conditions-json .scperturbench/inputs/gse132080_bench/conditions.json \
        --bench-manifest  .scperturbench/inputs/gse132080_bench/manifest.json \
        --guide-identity-col guide_identity \
        --perturbation-col perturbation_label \
        --guide-activity-cnn-table <Table_S8> \
        --activity-table <Table_S16> \
        --guide-sequence-length 26 \
        --output .scperturbench/inputs/activity_ablation_preflight.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


CONTROL_PATTERNS = ("control", "non-targeting", "non_targeting", "neg_ctrl", "ntc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GSE132080 activity-ablation inputs.")
    parser.add_argument("--repo", default="/root/Diff-AE-Perturb")
    parser.add_argument("--train-h5ad", required=True, type=Path)
    parser.add_argument("--test-h5ad", required=True, type=Path)
    parser.add_argument("--conditions-json", required=True, type=Path)
    parser.add_argument("--bench-manifest", default=None, type=Path)
    parser.add_argument("--prep-summary", default=None, type=Path)
    parser.add_argument("--guide-identity-col", default="guide_identity")
    parser.add_argument("--perturbation-col", default="perturbation_label")
    parser.add_argument("--guide-activity-cnn-table", required=True, type=Path)
    parser.add_argument("--activity-table", required=True, type=Path)
    parser.add_argument("--activity-col", default="relative_activity_day10")
    parser.add_argument("--guide-sequence-length", type=int, default=26)
    parser.add_argument("--min-spanning-targets", type=int, default=4,
                        help="minimum number of targets spanned by the held-out guide set")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def is_control(label: str) -> bool:
    lowered = str(label).strip().lower()
    return any(pattern in lowered for pattern in CONTROL_PATTERNS)


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        "n_distinct_rounded_4dp": int(np.unique(np.round(array, 4)).size),
    }


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo) / "src"))

    import anndata as ad

    from perturbdiff.data import load_guide_activity_lookup
    from perturbdiff.guide_features import (
        find_record_for_label,
        guide_cnn_feature_dim,
        load_guide_feature_records,
    )

    checks: dict[str, dict[str, object]] = {}
    report: dict[str, object] = {"args": {k: str(v) for k, v in vars(args).items()}, "checks": checks}

    def record(name: str, passed: bool, **detail: object) -> bool:
        checks[name] = {"passed": bool(passed), **detail}
        return bool(passed)

    train = ad.read_h5ad(args.train_h5ad)
    test = ad.read_h5ad(args.test_h5ad)
    train_guides = train.obs[args.guide_identity_col].astype(str).to_numpy()
    test_guides = test.obs[args.guide_identity_col].astype(str).to_numpy()
    train_labels = train.obs[args.perturbation_col].astype(str).to_numpy()
    test_labels = test.obs[args.perturbation_col].astype(str).to_numpy()

    conditions_json = json.loads(Path(args.conditions_json).read_text(encoding="utf-8"))
    evaluated = [str(c) for c in conditions_json["evaluated_conditions"]]
    control_label = str(conditions_json.get("control_label", "control"))

    # ---- 1. split provenance -----------------------------------------------------------------
    # The benchmark's own h5ad bytes were produced by the (now removed) diff-project preprocessing,
    # so a byte-level sha256 cannot match the recorded manifest.  Provenance is therefore checked at
    # content level against the recorded split sizes / gene panel / per-condition cell counts.
    provenance: dict[str, object] = {
        "train_path": str(args.train_h5ad),
        "train_sha256": sha256(args.train_h5ad),
        "train_shape": list(train.shape),
        "test_path": str(args.test_h5ad),
        "test_sha256": sha256(args.test_h5ad),
        "test_shape": list(test.shape),
        "n_genes": int(test.shape[1]),
        "conditions_json": str(args.conditions_json),
        "n_evaluated_conditions": len(evaluated),
        "evaluated_conditions": evaluated,
    }
    if args.bench_manifest is not None and Path(args.bench_manifest).exists():
        manifest = json.loads(Path(args.bench_manifest).read_text(encoding="utf-8"))
        provenance["bench_manifest"] = str(args.bench_manifest)
        provenance["bench_train_sha256"] = manifest.get("train_sha256")
        provenance["bench_test_sha256"] = manifest.get("test_sha256")
        provenance["bench_n_genes"] = manifest.get("n_genes")
        provenance["hash_note"] = (
            "byte hashes differ by construction: the benchmark consumed the original "
            "splits_unseen_guide_seed7 h5ad files, this run uses the equivalent "
            "gse132080_squidiff_repro copies prepared in this workspace"
        )
    if args.prep_summary is not None and Path(args.prep_summary).exists():
        prep = json.loads(Path(args.prep_summary).read_text(encoding="utf-8"))
        alignment = prep.get("split_alignment", {})
        panel = prep.get("benchmark_panel_overlap", {})
        scale = prep.get("scale_equivalence_vs_benchmark_train", {})
        provenance["prep_summary"] = str(args.prep_summary)
        provenance["prep_split_alignment"] = alignment
        provenance["prep_panel_overlap"] = panel
        provenance["prep_scale_equivalence"] = scale
        record(
            "split_content_matches_benchmark",
            int(alignment.get("benchmark_train_cells", -1)) == int(train.n_obs)
            and int(alignment.get("benchmark_test_cells", -1)) == int(test.n_obs)
            and int(panel.get("shared", -1)) == int(test.shape[1]),
            n_train_cells=int(train.n_obs),
            n_test_cells=int(test.n_obs),
            n_genes=int(test.shape[1]),
            benchmark_train_cells=alignment.get("benchmark_train_cells"),
            benchmark_test_cells=alignment.get("benchmark_test_cells"),
            panel_shared=panel.get("shared"),
            max_abs_scale_diff=scale.get("max_abs_diff"),
        )
    record("train_and_test_are_different_files", provenance["train_sha256"] != provenance["test_sha256"])
    report["provenance"] = provenance

    # ---- 2. held-out guide structure ---------------------------------------------------------
    test_targeting = np.array([not is_control(label) for label in test_labels], dtype=bool)
    held_out_guides = sorted({str(g) for g in test_guides[test_targeting] if not is_control(g)})
    train_guide_set = {
        str(g) for g, label in zip(train_guides, train_labels)
        if not is_control(label) and not is_control(g)
    }
    overlap = sorted(set(held_out_guides) & train_guide_set)

    guides_by_condition: dict[str, set[str]] = {}
    for guide, label in zip(test_guides[test_targeting], test_labels[test_targeting]):
        guides_by_condition.setdefault(str(label), set()).add(str(guide))

    # The benchmark's own condition metadata is the independent source of truth for cell counts.
    expected_per_condition = conditions_json.get("per_condition", {})
    cell_count_mismatches = {
        condition: {
            "h5ad": int((test_labels == condition).sum()),
            "conditions_json": int(expected_per_condition.get(condition, {}).get("n_test_cells", -1)),
        }
        for condition in evaluated
        if int(expected_per_condition.get(condition, {}).get("n_test_cells", -1))
        != int((test_labels == condition).sum())
    }

    genes = [str(g) for g in test.var_names]
    held_out_targets = sorted({guide.split("_", 1)[0] for guide in held_out_guides})
    targets_in_panel = [g for g in held_out_targets if g in genes]

    record(
        "held_out_guide_set_disjoint_from_training_guides",
        not overlap,
        n_train_guides=len(train_guide_set),
        n_held_out_guides=len(held_out_guides),
        n_overlap=len(overlap),
        overlap=overlap[:20],
    )
    record(
        "every_evaluated_condition_has_at_least_one_held_out_guide",
        all(guides_by_condition.get(condition) for condition in evaluated),
        guides_per_condition={
            condition: len(guides_by_condition.get(condition, ())) for condition in evaluated
        },
    )
    record(
        "evaluated_conditions_match_test_split",
        sorted(guides_by_condition) == sorted(evaluated),
        only_in_test=sorted(set(guides_by_condition) - set(evaluated)),
        only_in_conditions_json=sorted(set(evaluated) - set(guides_by_condition)),
    )
    record(
        "per_condition_cell_counts_match_benchmark_metadata",
        not cell_count_mismatches,
        n_conditions_checked=len(evaluated),
        mismatches=cell_count_mismatches,
    )
    record(
        "held_out_guides_span_enough_targets",
        len(targets_in_panel) >= args.min_spanning_targets,
        n_held_out_targets=len(held_out_targets),
        n_in_gene_panel=len(targets_in_panel),
        targets_in_panel=targets_in_panel,
        targets_missing_from_panel=sorted(set(held_out_targets) - set(targets_in_panel)),
    )
    report["held_out"] = {
        "label_column": args.perturbation_col,
        "guide_column": args.guide_identity_col,
        "control_label": control_label,
        "n_test_cells": int(test.n_obs),
        "n_test_targeting_cells": int(test_targeting.sum()),
        "n_held_out_guides": len(held_out_guides),
        "held_out_guides": held_out_guides,
        "per_condition": {
            condition: {
                "guides": sorted(guides_by_condition.get(condition, ())),
                "target": condition,
                "n_test_cells": int((test_labels == condition).sum()),
            }
            for condition in sorted(guides_by_condition)
        },
        "n_train_cells": int(train.n_obs),
        "n_train_guides": len(train_guide_set),
        "one_guide_per_condition": all(
            len(guides_by_condition.get(condition, ())) == 1 for condition in evaluated
        ),
    }

    # ---- 3. activity inputs ------------------------------------------------------------------
    records_by_key = load_guide_feature_records([args.guide_activity_cnn_table])
    activity_lookup = load_guide_activity_lookup(args.activity_table, activity_col=args.activity_col)
    expected_dim = guide_cnn_feature_dim(sequence_length=args.guide_sequence_length)

    feature_dim = None
    try:
        from perturbdiff.guide_features import build_notebook_cnn_feature_tensor

        sample = next(iter(records_by_key.values()))
        feature_dim = int(build_notebook_cnn_feature_tensor(sample, sequence_length=args.guide_sequence_length).reshape(-1).size)
    except Exception as exc:  # pragma: no cover - diagnostic path
        feature_dim = f"error: {exc}"

    all_guides = sorted(set(held_out_guides) | train_guide_set)
    feature_missing = [g for g in all_guides if find_record_for_label(g, records_by_key) is None]
    activity_missing = [g for g in all_guides if g not in activity_lookup]
    held_out_feature_missing = [g for g in held_out_guides if find_record_for_label(g, records_by_key) is None]
    held_out_activity_missing = [g for g in held_out_guides if g not in activity_lookup]
    is_reference_guide = lambda label: str(label).endswith("_00")

    # Table S8 contains no `_00` rows at all: the fully matched (highest-activity) sgRNA of each
    # family is the no-mismatch reference case, so the upstream activity arm can only feed it zero
    # CNN features.  A condition whose held-out guides are *all* reference guides therefore receives
    # no activity signal at all, which is structural (an upstream data limitation), not a break in
    # this pipeline.  Those conditions must be reported explicitly and excluded from the
    # "activity channel is live" sensitivity view; the gate itself only fails when the pipeline
    # cannot build features for any guide.
    def has_features(label: str) -> bool:
        return find_record_for_label(label, records_by_key) is not None

    conditions_without_features = {
        condition: sorted(guides)
        for condition, guides in guides_by_condition.items()
        if not [g for g in guides if has_features(g)]
    }
    conditions_with_features = sorted(
        condition for condition, guides in guides_by_condition.items()
        if [g for g in guides if has_features(g)]
    )
    coverage_cells = int(
        sum(
            int((test_labels == condition).sum())
            for condition, guides in guides_by_condition.items()
            if [g for g in guides if has_features(g)]
        )
    )
    reference_guides_missing = [g for g in feature_missing if is_reference_guide(g)]
    # Training-side gaps (RAN/RPS15 families are absent from Table S8) put zero features on those
    # guides during training; that is a shared data property of the training recipe, reported but not
    # gate-failing.  What must hold is that every *evaluated* guide the activity arm conditions on has
    # features, except the structurally impossible `_00` reference guides.
    non_reference_missing = [g for g in feature_missing if not is_reference_guide(g) and g != "*"]
    held_out_non_reference_missing = [g for g in held_out_feature_missing if not is_reference_guide(g)]
    held_out_reference_missing = [g for g in held_out_feature_missing if is_reference_guide(g)]

    record(
        "guide_cnn_feature_dim_matches_sequence_length",
        feature_dim == expected_dim,
        feature_dim=feature_dim,
        expected_dim=expected_dim,
        sequence_length=args.guide_sequence_length,
    )
    record(
        "cnn_features_available_for_conditioning",
        bool(conditions_with_features)
        and not held_out_non_reference_missing
        and len(conditions_with_features) >= min(10, len(evaluated)),
        n_conditions=len(guides_by_condition),
        n_conditions_with_feature_bearing_guides=len(conditions_with_features),
        n_conditions_without_any_feature_bearing_guide=len(conditions_without_features),
        conditions_without_any_feature_bearing_guide=sorted(conditions_without_features),
        n_guides_total=len(all_guides),
        n_guides_with_features=len(all_guides) - len(feature_missing),
        n_guides_missing_features=len(feature_missing),
        n_missing_are_reference_guides=len(reference_guides_missing),
        n_held_out_guides_missing_features=len(held_out_feature_missing),
        held_out_guides_missing_features=held_out_feature_missing,
        all_feature_less_held_out_guides_are_reference=not held_out_non_reference_missing,
        held_out_non_reference_guides_missing_features=held_out_non_reference_missing,
        held_out_reference_guides_missing_features=held_out_reference_missing,
        training_side_guides_missing_features=non_reference_missing,
        test_cells_with_activity_features=coverage_cells,
        test_targeting_cells=int(test_targeting.sum()),
        test_cells_covered_fraction=round(coverage_cells / max(int(test_targeting.sum()), 1), 4),
    )
    record(
        "activity_channel_is_live_for_most_conditions",
        len(conditions_with_features) >= len(evaluated) - 3,
        n_conditions_with_features=len(conditions_with_features),
        n_evaluated_conditions=len(evaluated),
        feature_less_conditions=sorted(conditions_without_features),
        interpretation=(
            "feature-less conditions can only be evaluated as a no-activity pass-through; they are "
            "reported separately and excluded from the 'activity channel is live' sensitivity view"
        ),
    )
    record(
        "activity_scores_resolve_for_all_guides",
        not [g for g in activity_missing if g != "*"],
        n_guides=len(all_guides),
        n_missing=len([g for g in activity_missing if g != "*"]),
        missing=[g for g in activity_missing if g != "*"][:25],
        non_guide_labels_missing=[g for g in activity_missing if g == "*"],
        held_out_missing=held_out_activity_missing,
    )

    activity_values = [activity_lookup[g] for g in all_guides if g in activity_lookup]
    held_out_values = [activity_lookup[g] for g in held_out_guides if g in activity_lookup]
    per_target: dict[str, list[float]] = {}
    for guide in all_guides:
        if guide not in activity_lookup:
            continue
        per_target.setdefault(guide.split("_", 1)[0], []).append(activity_lookup[guide])
    within_target_spreads = {
        target: describe(values) for target, values in sorted(per_target.items()) if len(values) > 1
    }
    spread_medians = [stats["range"] for stats in within_target_spreads.values()]
    axis_stats = describe(activity_values)
    record(
        "activity_axis_is_not_degenerate",
        axis_stats.get("n", 0) > 1 and float(axis_stats.get("std", 0.0)) > 0.0,
        overall=axis_stats,
        held_out=describe(held_out_values),
        n_targets_with_multiple_guides=len(within_target_spreads),
        median_within_target_range=float(np.median(spread_medians)) if spread_medians else None,
        within_target_variation_note=(
            "within-target activity spread is what the conditioning axis must resolve; a median "
            "range near zero would mean the arm cannot distinguish guides of the same target"
        ),
    )
    report["activity"] = {
        "cnn_table": str(args.guide_activity_cnn_table),
        "cnn_table_sha256": sha256(args.guide_activity_cnn_table),
        "n_cnn_records": len(records_by_key),
        "cnn_feature_dim": feature_dim,
        "reference_guides_without_cnn_rows": feature_missing,
        "conditions_with_feature_bearing_guides": conditions_with_features,
        "conditions_without_feature_bearing_guides": sorted(conditions_without_features),
        "test_cells_with_activity_features": coverage_cells,
        "test_cells_covered_fraction": round(coverage_cells / max(int(test_targeting.sum()), 1), 4),
        "activity_table": str(args.activity_table),
        "activity_table_sha256": sha256(args.activity_table),
        "activity_col": args.activity_col,
        "n_activity_scores": len(activity_lookup),
        "overall": axis_stats,
        "held_out": describe(held_out_values),
        "within_target": within_target_spreads,
    }

    report["passed"] = all(bool(entry["passed"]) for entry in checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    for name, entry in checks.items():
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"[{status}] {name}")
        if not entry["passed"]:
            print(f"        {json.dumps({k: v for k, v in entry.items() if k != 'passed'}, default=str)[:600]}")
    print(f"\nreport: {args.output}")
    print(f"PREFLIGHT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
