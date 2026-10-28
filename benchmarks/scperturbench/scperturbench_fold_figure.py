"""Six-metric figure for a SINGLE knockdown-stratum band.

Reuses ``scperturbench_ablation_figure`` verbatim - the same module that produced
the GSE132080 figures - so panel layout, bootstrap CIs, paired overlays, effect
sizes and every annotation match that style.  The only difference is which rows go
in: this selects one fold instead of pooling all folds.

Scope and honest labelling
--------------------------
A band is 10-12 guides, so this is a *descriptive* view of one stratum, not a
pre-registered test.  The figure says so in its subtitle, and the statistics are
reported without BH correction (correcting across six metrics over a single
n=11 band would neither be pre-registered nor meaningful).  The pooled
out-of-fold analysis remains the headline.

Usage::

    python scperturbench_fold_figure.py \
        --fold 1 --state esc --arm-on activity_on --arm-off activity_off
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

ARM_LABELS = {
    "activity_on": "measured screen activity",
    "activity_off": "measured screen activity",
    "activity_mask": "measurement mask",
    "activity_perm": "permuted activity",
}

STRATUM_NOTE = {
    0: "low-knockdown band",
    1: "mid-knockdown band",
    2: "high-knockdown band",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Six-metric figure for one stratum.")
    parser.add_argument("--results-root", type=Path,
                        default=HERE / ".scperturbench/results/gse283614_activity_ablation_kd3")
    parser.add_argument("--inputs-dir", type=Path,
                        default=HERE / ".scperturbench/inputs/gse283614_kd3_inputs")
    parser.add_argument("--split-dir", type=Path,
                        default=HERE / ".scperturbench/inputs/gse283614_knockdown3")
    parser.add_argument("--dosage-table", type=Path,
                        default=Path("/root/Diff-AE-Perturb/reproducibility/prepared/gse283614/dose_cv/gse283614_tf_dosage.tsv"))
    parser.add_argument("--dataset", default="gse283614")
    parser.add_argument("--state", default="esc")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arm-on", default="activity_on")
    parser.add_argument("--arm-off", default="activity_off")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--tag", default=None, help="Filename tag; defaults to the arm names.")
    parser.add_argument("--minimal", action="store_true",
                        help="Strip the subtitle, the parenthetical budget note and every statistics "
                             "block; use one-line 'with activity' / 'without activity' labels.")
    parser.add_argument("--title", default=None, help="Override the figure title.")
    return parser.parse_args()


def load_fold(results_root: Path, dataset: str, state: str, fold: int, arm: str, seed: int) -> pd.DataFrame:
    path = results_root / dataset / state / f"fold_{fold}" / arm / f"seed{seed}" / "score" / "per_condition.csv"
    if not path.exists():
        raise SystemExit(f"missing per-condition scores: {path}")
    frame = pd.read_csv(path)
    frame["arm"] = arm
    frame["fold"] = fold
    return frame


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (
        args.results_root / args.dataset / args.state / f"fold_{args.fold}_figure"
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    conditions = json.loads(
        (args.inputs_dir / f"conditions_{args.state}.json").read_text(encoding="utf-8")
    )
    pc = conditions["per_condition"]
    fold_guides = sorted(
        g for g in conditions["evaluated_conditions"] if pc[g]["fold"] == args.fold
    )

    on = load_fold(args.results_root, args.dataset, args.state, args.fold, args.arm_on, args.seed)
    off = load_fold(args.results_root, args.dataset, args.state, args.fold, args.arm_off, args.seed)
    frame = pd.concat([off, on], ignore_index=True)

    scored = set(frame["condition"].unique())
    missing = [g for g in fold_guides if g not in scored]
    present = [g for g in fold_guides if g in scored]
    if missing:
        print(
            f"[figure] note: {len(missing)} guide(s) not scored in fold {args.fold} "
            f"(dropped for both arms): {missing}"
        )
    if len(present) < 5:
        raise SystemExit(f"too few paired guides in fold {args.fold}: {len(present)}")
    frame = frame[frame["condition"].isin(present)]

    # Knudown range of the band, for the subtitle.
    dosage = pd.read_csv(args.dosage_table, sep="\t")
    dose = dict(zip(dosage["guide_identity"].astype(str), dosage["knockdown_fraction"]))
    target = dict(zip(dosage["guide_identity"].astype(str), dosage["dose_target"]))
    parts = []
    for t in ("NANOG", "OCT4"):
        vals = sorted(dose[g] for g in present if target.get(g) == t and g in dose)
        if vals:
            parts.append(f"{t} {vals[0]:.2f}-{vals[-1]:.2f}")
    band_desc = ", ".join(parts)
    n_nonzero = sum(1 for g in present if pc[g]["activity_input"] > 0)

    rng = np.random.default_rng(base.BOOTSTRAP_SEED)
    all_metrics = base.PRIMARY_METRICS + base.SECONDARY_METRICS
    pairwise, tests, long_frame = base.build_analysis(
        frame, args.arm_on, args.arm_off, rng, all_metrics
    )

    # provenance attached to each guide, so the CSV explains itself
    meta = pd.DataFrame(
        [
            {
                "condition": g,
                "fold": args.fold,
                "target_gene": pc[g]["target_gene"],
                "guide_type": pc[g]["guide_type"],
                "activity_input": pc[g]["activity_input"],
                "activity_measured": pc[g]["activity_measured"],
                "knockdown_fraction": dose.get(g, np.nan),
                "n_test_cells": pc[g]["n_test_cells"],
            }
            for g in present
        ]
    )
    long_frame = long_frame.merge(meta, on="condition", how="left")

    pairwise.to_csv(output_dir / "fold_pairwise.csv", index=False)
    tests.to_csv(output_dir / "fold_tests.csv", index=False)
    long_frame.to_csv(output_dir / "fold_per_condition.csv", index=False)
    meta.to_csv(output_dir / "fold_guides.csv", index=False)

    state_upper = args.state.upper()
    stratum = STRATUM_NOTE.get(args.fold, f"stratum {args.fold}")
    subtitle = (
        f"{state_upper} {stratum} held out as one contiguous block; "
        f"n={len(present)} paired guides; band knockdown {band_desc}; "
        f"{n_nonzero} of {len(present)} guides carry non-zero activity; "
        f"error bars = 95% bootstrap CI over guides; seed {args.seed}; DDIM 50; "
        f"single-band descriptive view, not BH-corrected"
    )
    title = args.title or (
        f"GSE283614 {state_upper} · {stratum} · "
        f"{ARM_LABELS.get(args.arm_on, args.arm_on)} conditioning"
    )
    if args.minimal and args.title is None:
        # Short, label-free title for the stripped-down variant.
        title = f"GSE283614 {state_upper} · {stratum}"

    tag = args.tag or f"{args.arm_on}_vs_{args.arm_off}"
    base.plot_metrics(
        long_frame,
        pairwise,
        metrics=base.PRIMARY_METRICS,
        arm_on=args.arm_on,
        arm_off=args.arm_off,
        title=title,
        out_base=figures_dir / f"fig_fold{args.fold}_{tag}_6metrics",
        dpi=args.dpi,
        subtitle=subtitle,
        arm_on_label=ARM_LABELS.get(args.arm_on, args.arm_on),
        arm_off_label=ARM_LABELS.get(args.arm_off, args.arm_off),
        minimal=args.minimal,
    )

    print(f"\n=== {state_upper} fold{args.fold} ({stratum}) · {args.arm_on} vs {args.arm_off} · n={len(present)} ===")
    # `tests` carries only the test statistics; the arm means live in `pairwise`.
    summary = tests.merge(
        pairwise[
            [
                "metric",
                "deg_budget",
                "mean_activity_on",
                "mean_activity_off",
                "relative_change_pct",
            ]
        ],
        on=["metric", "deg_budget"],
        how="left",
    )
    view = summary[summary["deg_budget"].isin(["100", "cutoff"])]
    view = view[view["metric"].isin([m for m, _, _, _ in base.PRIMARY_METRICS])]
    print(
        view[
            [
                "metric",
                "mean_activity_off",
                "mean_activity_on",
                "relative_change_pct",
                "wilcoxon_p",
                "paired_t_p",
                "hedges_g_paired",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4g}")
    )
    print(f"\nfigures + tables: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
