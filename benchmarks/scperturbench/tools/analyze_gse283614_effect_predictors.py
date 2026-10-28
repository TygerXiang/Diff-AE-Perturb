"""Correlate the per-guide activity effect with quantity of the activity signal.

The GSE132080 report could not explain why its effect concentrated in a few
conditions: 16 Spearman correlations between the per-condition effect and
"how much activity information this target has" were all null.  This tool runs
the analogous analysis for GSE283614 and, more importantly, answers the question
that analysis could not:

* does the per-guide effect of activity conditioning track |activity| itself?
  (``rho_effect_vs_abs_activity``)
* does it track the *displacement* the activity channel actually produces in
  prediction space?  (``rho_effect_vs_probe_shift_between``).  If the benefit is
  real, guides whose predictions the channel moves most should be the guides whose
  accuracy changes most.
* does it track whether the guide was measured at all, i.e. the binary mask?
  (``rho_effect_vs_measured``)

Inputs are the analysis CSVs plus the A2 probe output; nothing is recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HIGHER_IS_BETTER = {
    "mse": False,
    "pcc_delta": True,
    "e_distance": False,
    "wasserstein": False,
    "kl_divergence": False,
    "common_degs": True,
}
PRIMARY = [("mse", "100"), ("pcc_delta", "100"), ("e_distance", "100"),
           ("wasserstein", "100"), ("kl_divergence", "100"), ("common_degs", "cutoff")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain the per-guide activity effect.")
    parser.add_argument("--analysis", type=Path, required=True,
                        help="ablation_analysis directory for one state.")
    parser.add_argument("--probe", type=Path, default=None,
                        help="Directory holding fold_*/activity_on/activity_perturbation_probe_seed*.csv.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_probe(probe_dir: Path | None) -> pd.DataFrame:
    if probe_dir is None:
        return pd.DataFrame()
    frames = []
    for path in sorted(probe_dir.glob("fold_*/activity_on/activity_perturbation_probe_seed*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    probe = pd.concat(frames, ignore_index=True)
    probe = probe[probe["activity_delta"].abs() > 1e-12]
    return probe.set_index("guide")


def main() -> int:
    args = parse_args()
    per_condition = pd.read_csv(args.analysis / "ablation_per_condition.csv")
    probe = load_probe(args.probe)
    rows = []
    for metric, budget in PRIMARY:
        subset = per_condition[
            (per_condition["metric"] == metric) & (per_condition["deg_budget"].astype(str) == budget)
        ].copy()
        if subset.empty:
            continue
        direction = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
        # Signed improvement: positive always means "the activity arm is better".
        subset["improvement"] = direction * subset["paired_delta_on_minus_off"]
        subset = subset.set_index("condition")
        if not probe.empty:
            subset = subset.join(
                probe[["activity_true", "activity_swapped", "l2_shift_swapped", "ratio_shift_over_between_guide"]],
                how="left",
            )
        finite = subset[np.isfinite(subset["improvement"])]

        def rho(column: str) -> tuple[float, float, int]:
            if column not in finite.columns:
                return float("nan"), float("nan"), 0
            pair = finite[[column, "improvement"]].dropna()
            pair = pair[np.isfinite(pair[column])]
            if len(pair) < 4 or pair[column].nunique() < 2:
                return float("nan"), float("nan"), int(len(pair))
            value, p = stats.spearmanr(pair[column], pair["improvement"])
            return float(value), float(p), int(len(pair))

        r_abs, p_abs, n_abs = rho("activity_input")
        r_shift, p_shift, n_shift = rho("ratio_shift_over_between_guide")
        measured = finite.assign(measured=finite["activity_measured"].astype(float))
        if measured["measured"].nunique() == 2:
            r_mask, p_mask = stats.spearmanr(measured["measured"], measured["improvement"])
        else:
            r_mask, p_mask = float("nan"), float("nan")
        rows.append(
            {
                "metric": metric,
                "deg_budget": budget,
                "n_guides": int(len(finite)),
                "mean_improvement": float(finite["improvement"].mean()),
                "n_guides_improved": int((finite["improvement"] > 0).sum()),
                "rho_effect_vs_abs_activity": r_abs,
                "p_effect_vs_abs_activity": p_abs,
                "n_effect_vs_abs_activity": n_abs,
                "rho_effect_vs_probe_shift_between": r_shift,
                "p_effect_vs_probe_shift_between": p_shift,
                "n_effect_vs_probe_shift_between": n_shift,
                "rho_effect_vs_measured": float(r_mask),
                "p_effect_vs_measured": float(p_mask),
            }
        )
        # Per-target breakdown: between-target spread is library design, so only
        # within-target correlations can speak to "more activity information
        # helped more".
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    by_target = []
    for metric, budget in PRIMARY:
        subset = per_condition[
            (per_condition["metric"] == metric) & (per_condition["deg_budget"].astype(str) == budget)
        ].copy()
        subset["improvement"] = (1.0 if HIGHER_IS_BETTER[metric] else -1.0) * subset[
            "paired_delta_on_minus_off"
        ]
        for target, group in subset.groupby("target"):
            group = group[np.isfinite(group["improvement"])]
            if len(group) < 4 or group["activity_input"].nunique() < 2:
                continue
            r, p = stats.spearmanr(group["activity_input"], group["improvement"])
            by_target.append(
                {
                    "metric": metric,
                    "deg_budget": budget,
                    "target": target,
                    "n_guides": int(len(group)),
                    "spearman_rho": float(r),
                    "p_value": float(p),
                }
            )
    if by_target:
        pd.DataFrame(by_target).to_csv(args.output.with_name("effect_by_target.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
