"""Pooled out-of-fold paired analysis for the GSE283614 measured-activity ablation.

Deliberately a *driver*, not a reimplementation: the paired statistics, effect
sizes, BH correction and every figure come from ``scperturbench_ablation_figure``,
the same module the GSE132080 report used.  Only data loading and figure captions
differ, because this design is a 5-fold out-of-fold protocol rather than a single
split.

Layout expected under ``--results-root`` (``.scperturbench/results/gse283614_activity_ablation``)::

    gse283614/<state>/fold_<k>/<arm>/seed<seed>/score/per_condition.csv

Every guide is held out exactly once across the five folds (asserted by the
preflight), so concatenating the folds yields one prediction per guide and the
pairing unit stays the *guide*.

Outputs (under ``--output-dir``):

    ablation_per_condition.csv          OOF per-guide values for both arms, with fold/type/activity
    ablation_pairwise.csv               arm means, paired deltas, bootstrap CIs
    ablation_tests.csv                  Wilcoxon + paired t, Hedges g, BH q
    ablation_summary_wide.csv           2 arms x metric/budget means
    ablation_subsets.csv                the pre-registered subset tests
    activity_coverage.csv              per-guide activity value and measured/unmeasured status
    figures/fig1_activity_ablation_6metrics.{png,pdf}
    figures/fig2_..._activity_live.{png,pdf}      (measured-activity guides only)
    figures/fig3_per_condition_deltas.{png,pdf}

Statistics policy
-----------------
* Primary metrics are the six DEG-100/cutoff cells, identical to GSE132080.
* The denominator for the headline test is **all evaluated guides** (n = 33 in
  ESC); restricting to measured-activity guides is a pre-registered sensitivity
  analysis, not the headline, because it halves n and because the excluded guides
  are exactly the promoter guides (a structural, not random, exclusion).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import scperturbench_ablation_figure as base  # noqa: E402

ARM_ON = "activity_on"
ARM_OFF = "activity_off"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse the GSE283614 measured-activity ablation.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=HERE / ".scperturbench/results/gse283614_activity_ablation",
    )
    parser.add_argument("--dataset", default="gse283614")
    parser.add_argument("--state", default="esc")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arm-on", default=ARM_ON)
    parser.add_argument("--arm-off", default=ARM_OFF)
    parser.add_argument("--inputs-dir", type=Path, default=None,
                        help="Inputs directory holding conditions_<state>.json (default: the dose_cv one).")
    parser.add_argument("--split-tag", default="",
                        help="Label of the fold set, recorded in the report header (e.g. kd3).")
    parser.add_argument("--conditions-json", type=Path, default=None,
                        help="conditions_<state>.json from the input builder.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="GSE283614 · measured sgRNA-activity conditioning")
    parser.add_argument("--allow-missing-conditions", action="store_true", default=True,
                        help="Score the intersection of pre-registered and actually-scored guides "
                             "instead of failing. Guides can be dropped by the adapter's "
                             "--min-test-cells in a fold whose blocks are smaller.")
    return parser.parse_args()


def load_arm_oof(
    results_root: Path, dataset: str, state: str, arm: str, seed: int
) -> tuple[pd.DataFrame, list[int]]:
    """Concatenate one arm's per-fold per-condition scores."""

    fold_dirs = sorted((results_root / dataset / state).glob("fold_*"))
    if not fold_dirs:
        raise SystemExit(f"no fold directories under {results_root / dataset / state}")
    frames, used = [], []
    for fold_dir in fold_dirs:
        path = fold_dir / arm / f"seed{seed}" / "score" / "per_condition.csv"
        if not path.exists():
            print(f"  WARNING: missing {path}; skipping this fold for {arm}")
            continue
        frame = pd.read_csv(path)
        frame["arm"] = arm
        frame["fold"] = int(fold_dir.name.split("_")[1])
        frames.append(frame)
        used.append(int(fold_dir.name.split("_")[1]))
    if not frames:
        raise SystemExit(f"no per-condition scores for {arm}")
    pooled = pd.concat(frames, ignore_index=True)
    # A guide must appear in exactly one fold; verify rather than assume.
    dupes = (
        pooled[pooled["arm"] == arm]
        .groupby("condition")["fold"]
        .nunique()
        .loc[lambda s: s > 1]
    )
    if not dupes.empty:
        raise SystemExit(f"{arm}: conditions scored in more than one fold: {dupes.to_dict()}")
    return pooled, used


def build_subsets(
    conditions: dict, state: str
) -> dict[str, dict[str, list[str]]]:
    """Pre-registered subset definitions for the sensitivity analyses."""

    per_condition = conditions["per_condition"]
    evaluated = list(conditions["evaluated_conditions"])
    measured = [g for g in evaluated if per_condition[g]["activity_measured"]]
    unmeasured = [g for g in evaluated if not per_condition[g]["activity_measured"]]
    promoter = [g for g in evaluated if per_condition[g]["guide_type"] == "promoter"]
    enhancer = [g for g in evaluated if per_condition[g]["guide_type"] == "enhancer"]
    subsets: dict[str, dict[str, list[str]]] = {
        "all_evaluated": {"conditions": evaluated},
        "measured_activity": {"conditions": measured},
        "unmeasured_activity_zero_filled": {"conditions": unmeasured},
        "promoter": {"conditions": promoter},
        "enhancer": {"conditions": enhancer},
    }
    for target in ("NANOG", "OCT4"):
        subsets[f"target_{target}"] = {
            "conditions": [
                g
                for g in evaluated
                if base_target(per_condition[g]["target_gene"]) == target
            ]
        }
    return subsets


def base_target(target_gene: str) -> str:
    return {"NANOG": "NANOG", "POU5F1": "OCT4"}.get(str(target_gene), str(target_gene))


# Human-readable channel names.  The GSE283614 axis is a *measured* screen
# depletion, not a predicted efficiency, so the figure must not claim otherwise.
ARM_LABELS = {
    "activity_on": "measured screen activity",
    "activity_off": "measured screen activity",
    "activity_perm": "permuted activity",
    "activity_mask": "measurement mask",
}


def arm_label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm)


def main() -> int:
    args = parse_args()
    results_root = args.results_root
    output_dir = args.output_dir or results_root / args.dataset / args.state / "ablation_analysis"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    conditions_path = args.conditions_json or (
        (args.inputs_dir or HERE / ".scperturbench/inputs/gse283614_activity_ablation")
        / f"conditions_{args.state}.json"
    )
    conditions = json.loads(Path(conditions_path).read_text(encoding="utf-8"))
    per_condition_meta = conditions["per_condition"]
    if args.split_tag:
        print(f"[analysis] fold set: {args.split_tag}  conditions: {conditions_path}")

    off_frame, folds = load_arm_oof(results_root, args.dataset, args.state, args.arm_off, args.seed)
    on_frame, folds_on = load_arm_oof(results_root, args.dataset, args.state, args.arm_on, args.seed)
    if sorted(folds) != sorted(folds_on):
        raise SystemExit(f"arms cover different folds: {sorted(folds)} vs {sorted(folds_on)}")
    frame = pd.concat([off_frame, on_frame], ignore_index=True)

    evaluated = set(conditions["evaluated_conditions"])
    scored = set(frame["condition"].unique())
    # The adapter drops conditions below --min-test-cells, and a small fold block
    # can push a guide under that threshold.  Both arms lose the same guides, so
    # the pairing stays intact; score the intersection and say what was dropped
    # rather than refusing to analyse.
    only_in_scores = sorted(scored - evaluated)
    dropped = sorted(evaluated - scored)
    if only_in_scores:
        raise SystemExit(f"scores contain conditions that were never pre-registered: {only_in_scores}")
    if dropped:
        detail = {
            g: per_condition_meta.get(g, {}).get("n_test_cells", "?") for g in dropped
        }
        print(
            f"[analysis] WARNING: {len(dropped)} pre-registered guide(s) were not scored "
            f"(most likely below --min-test-cells in their fold); dropped for BOTH arms: {detail}"
        )
        (output_dir / "dropped_conditions.json").write_text(
            json.dumps(
                {
                    "dropped": dropped,
                    "reason": "not scored by the adapter in any fold; identical for both arms",
                    "per_guide_test_cells": detail,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    rng = np.random.default_rng(base.BOOTSTRAP_SEED)
    all_metrics = base.PRIMARY_METRICS + base.SECONDARY_METRICS
    pairwise, tests, long_frame = base.build_analysis(
        frame, args.arm_on, args.arm_off, rng, all_metrics
    )
    tests = tests.merge(
        pairwise[
            [
                "metric", "deg_budget", "label", "higher_is_better",
                "mean_activity_on", "mean_activity_off",
                "mean_paired_delta_on_minus_off", "relative_change_pct",
            ]
        ],
        on=["metric", "deg_budget"],
        how="left",
    )

    # Attach per-guide provenance so the report can split groups without re-reading JSON.
    meta = pd.DataFrame(
        [
            {
                "condition": guide,
                "fold": per_condition_meta[guide]["fold"],
                "target_gene": per_condition_meta[guide]["target_gene"],
                "target": base_target(per_condition_meta[guide]["target_gene"]),
                "guide_type": per_condition_meta[guide]["guide_type"],
                "activity_measured": per_condition_meta[guide]["activity_measured"],
                "activity_input": per_condition_meta[guide]["activity_input"],
                "n_test_cells": per_condition_meta[guide]["n_test_cells"],
            }
            for guide in conditions["evaluated_conditions"]
        ]
    )
    long_frame = long_frame.merge(meta, on="condition", how="left")
    long_frame["paired_delta_on_minus_off"] = (
        long_frame["value_activity_on"] - long_frame["value_activity_off"]
    )

    pairwise.to_csv(output_dir / "ablation_pairwise.csv", index=False)
    tests.to_csv(output_dir / "ablation_tests.csv", index=False)
    long_frame.to_csv(output_dir / "ablation_per_condition.csv", index=False)

    wide = long_frame.copy()
    wide["metric_budget"] = wide["metric"] + "@" + wide["deg_budget"]
    wide_summary = (
        wide.groupby(["metric_budget", "higher_is_better"])[
            ["value_activity_on", "value_activity_off"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "value_activity_on": f"mean_{args.arm_on}",
                "value_activity_off": f"mean_{args.arm_off}",
            }
        )
    )
    wide_summary["mean_paired_delta_on_minus_off"] = (
        wide_summary[f"mean_{args.arm_on}"] - wide_summary[f"mean_{args.arm_off}"]
    )
    wide_summary.to_csv(output_dir / "ablation_summary_wide.csv", index=False)

    meta.to_csv(output_dir / "activity_coverage.csv", index=False)

    # ---- pre-registered subset tests -------------------------------------------------------
    subsets = build_subsets(conditions, args.state)
    subset_rows: list[dict] = []
    for name, spec in subsets.items():
        members = sorted(set(spec["conditions"]) & scored)
        if len(members) < 3:
            subset_rows.append({"subset": name, "n_conditions": len(members), "note": "too few to test"})
            continue
        sub_frame = frame[frame["condition"].isin(members)]
        sub_rng = np.random.default_rng(base.BOOTSTRAP_SEED + 7)
        sub_pairwise, sub_tests, _ = base.build_analysis(
            sub_frame, args.arm_on, args.arm_off, sub_rng, base.PRIMARY_METRICS
        )
        for _, row in sub_tests.iterrows():
            pair = sub_pairwise[
                (sub_pairwise["metric"] == row["metric"])
                & (sub_pairwise["deg_budget"] == row["deg_budget"])
            ]
            pair = pair.iloc[0] if not pair.empty else None
            subset_rows.append(
                {
                    "subset": name,
                    "n_conditions": int(row["n_conditions"]),
                    "metric": row["metric"],
                    "deg_budget": row["deg_budget"],
                    "mean_activity_on": None if pair is None else pair["mean_activity_on"],
                    "mean_activity_off": None if pair is None else pair["mean_activity_off"],
                    "mean_paired_delta": None if pair is None else pair["mean_paired_delta_on_minus_off"],
                    "relative_change_pct": None if pair is None else pair["relative_change_pct"],
                    "wilcoxon_p": row["wilcoxon_p"],
                    "wilcoxon_q_bh": row["wilcoxon_q_bh"],
                    "paired_t_p": row["paired_t_p"],
                    "hedges_g_paired": row["hedges_g_paired"],
                    "n_conditions_on_better": None if pair is None else pair["n_conditions_activity_on_better"],
                }
            )
    pd.DataFrame(subset_rows).to_csv(output_dir / "ablation_subsets.csv", index=False)

    # ---- figures ---------------------------------------------------------------------------
    n_conditions = int(long_frame[long_frame["deg_budget"] == "100"]["condition"].nunique())
    n_folds = int(long_frame["fold"].nunique())
    fold_design = (
        "knockdown-stratum blocks (each held-out set is one strength band, never trained on)"
        if args.split_tag == "kd3"
        else "rank-stratified folds (each held-out set spans the full knockdown range)"
    )
    subtitle = (
        f"out-of-fold over {n_folds} {fold_design}; n={n_conditions} guide conditions; "
        f"error bars = 95% bootstrap CI over guides; seed {args.seed}; DDIM 50; "
        f"same folds and recipe, only --activity-source differs"
    )
    base.plot_metrics(
        long_frame,
        pairwise,
        metrics=base.PRIMARY_METRICS,
        arm_on=args.arm_on,
        arm_off=args.arm_off,
        title=args.title,
        out_base=figures_dir / "fig1_activity_ablation_6metrics",
        dpi=args.dpi,
        subtitle=subtitle,
        arm_on_label=arm_label(args.arm_on),
        arm_off_label=arm_label(args.arm_off),
    )
    base.plot_per_condition_deltas(
        long_frame,
        metrics=base.PRIMARY_METRICS,
        arm_on=args.arm_on,
        arm_off=args.arm_off,
        title=f"{args.title} · per-guide effect",
        out_base=figures_dir / "fig3_per_condition_deltas",
        dpi=args.dpi,
        live_conditions=subsets["measured_activity"]["conditions"],
    )

    live = sorted(set(subsets["measured_activity"]["conditions"]) & scored)
    live_pairwise, live_tests, _ = base.build_analysis(
        frame[frame["condition"].isin(live)],
        args.arm_on,
        args.arm_off,
        np.random.default_rng(base.BOOTSTRAP_SEED + 1),
        base.PRIMARY_METRICS,
    )
    live_pairwise.to_csv(output_dir / "ablation_pairwise_measured_activity.csv", index=False)
    live_tests.to_csv(output_dir / "ablation_tests_measured_activity.csv", index=False)
    base.plot_metrics(
        long_frame[long_frame["condition"].isin(live)],
        live_pairwise,
        metrics=base.PRIMARY_METRICS,
        arm_on=args.arm_on,
        arm_off=args.arm_off,
        title=f"{args.title} · measured-activity guides only",
        out_base=figures_dir / "fig2_activity_ablation_6metrics_measured_only",
        dpi=args.dpi,
        arm_on_label=arm_label(args.arm_on),
        arm_off_label=arm_label(args.arm_off),
        subtitle=(
            f"restricted to the {len(live)} guides with a measured screen value; excluded "
            f"(unmeasured, written as 0): {', '.join(sorted(set(subsets['unmeasured_activity_zero_filled']['conditions']) & scored)) or 'none'}"
        ),
    )

    # ---- console summary -------------------------------------------------------------------
    print(f"\n=== {args.state.upper()} primary metrics (all {n_conditions} guide conditions, OOF) ===")
    view = tests[tests["deg_budget"].isin(["100", "cutoff"])].copy()
    view = view[view["metric"].isin([m for m, _, _, _ in base.PRIMARY_METRICS])]
    cols = [
        "metric", "mean_activity_off", "mean_activity_on", "mean_paired_delta_on_minus_off",
        "relative_change_pct", "wilcoxon_p", "wilcoxon_q_bh", "paired_t_p", "hedges_g_paired",
    ]
    print(view[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nanalysis artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
