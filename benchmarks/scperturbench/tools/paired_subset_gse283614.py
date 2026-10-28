"""Paired arm comparison on an arbitrary pre-registered guide subset.

The mask control (A3b) needs one comparison the figure driver cannot express:
``activity_on`` restricted to the guides whose value is strictly positive, against
``activity_mask`` on the same guides.  Extending the mask from "measured" to "1.0"
is only a real change where the value was non-zero, so that subset is exactly what
the pair of arms should be compared on.

Running this from the per-condition CSVs rather than retraining keeps the GPU free
and guarantees the numbers come from the same scored predictions as everything
else.

Statistics are imported from ``scperturbench_ablation_figure`` so the effect
sizes and BH correction are identical to the main analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
for candidate in (WORKSPACE, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import scperturbench_ablation_figure as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison on a guide subset.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", default="gse283614")
    parser.add_argument("--state", default="esc")
    parser.add_argument("--arm-on", required=True)
    parser.add_argument("--arm-off", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--subset", choices=("all", "positive_activity", "measured", "unmeasured"), default="all")
    parser.add_argument("--conditions-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def load_oof(results_root: Path, dataset: str, state: str, arm: str, seed: int) -> pd.DataFrame:
    """Concatenate an arm's per-fold per-condition scores, with the arm label."""

    frames = []
    for fold_dir in sorted((results_root / dataset / state).glob("fold_*")):
        path = fold_dir / arm / f"seed{seed}" / "score" / "per_condition.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["arm"] = arm
        frame["fold"] = int(fold_dir.name.split("_")[1])
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no per-condition scores for {arm}")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    conditions = json.loads(Path(args.conditions_json).read_text(encoding="utf-8"))
    per_condition = conditions["per_condition"]

    if args.subset == "all":
        members = list(conditions["evaluated_conditions"])
    elif args.subset == "positive_activity":
        members = [g for g in conditions["evaluated_conditions"] if per_condition[g]["activity_input"] > 0]
    elif args.subset == "measured":
        members = [g for g in conditions["evaluated_conditions"] if per_condition[g]["activity_measured"]]
    else:
        members = [g for g in conditions["evaluated_conditions"] if not per_condition[g]["activity_measured"]]

    frame = pd.concat(
        [
            load_oof(args.results_root, args.dataset, args.state, args.arm_off, args.seed),
            load_oof(args.results_root, args.dataset, args.state, args.arm_on, args.seed),
        ],
        ignore_index=True,
    )
    scored = set(frame["condition"].unique())
    missing = [g for g in members if g not in scored]
    if missing:
        raise SystemExit(f"subset members not scored: {missing[:5]}")
    sub = frame[frame["condition"].isin(members)]

    pairwise, tests, long_frame = base.build_analysis(
        sub,
        args.arm_on,
        args.arm_off,
        np.random.default_rng(base.BOOTSTRAP_SEED),
        base.PRIMARY_METRICS + base.SECONDARY_METRICS,
    )
    tests = tests.merge(
        pairwise[
            [
                "metric", "deg_budget", "label", "higher_is_better",
                "mean_activity_on", "mean_activity_off",
                "mean_paired_delta_on_minus_off", "relative_change_pct",
                "n_conditions_activity_on_better",
            ]
        ],
        on=["metric", "deg_budget"],
        how="left",
    )
    # ``label`` and ``n_conditions`` already arrive from build_analysis/merge, so
    # the identifying columns added here use distinct names.
    tests.insert(0, "subset", args.subset)
    tests.insert(0, "comparison_label", args.label or f"{args.arm_on}_vs_{args.arm_off}")
    tests.insert(0, "state", args.state)
    tests.to_csv(args.output, index=False)
    long_frame.to_csv(args.output.with_name(args.output.stem + "_per_condition.csv"), index=False)

    print(f"=== {args.state} {args.arm_on} vs {args.arm_off} on {args.subset} (n={len(members)}) ===")
    view = tests[tests["deg_budget"].isin(["100", "cutoff"])]
    cols = ["metric", "n_conditions", "mean_activity_off", "mean_activity_on",
            "mean_paired_delta_on_minus_off", "relative_change_pct",
            "wilcoxon_p", "wilcoxon_q_bh", "paired_t_p", "hedges_g_paired"]
    print(view[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
