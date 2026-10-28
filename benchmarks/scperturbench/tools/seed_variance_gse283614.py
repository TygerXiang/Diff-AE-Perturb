"""A4 seed-variance analysis for the GSE283614 ablation (host-side).

The GSE132080 report's clearest weakness was single-seed evidence: its uncertainty
came from between-condition structure, not from training randomness.  This reports
which source the ESC activity-arm result actually lives in.

For every guide, the seed-7 / seed-17 / seed-27 arms give three paired
observations.  Two variances matter:

* within-guide, between-seed SD  -> training randomness
* SD of the per-seed means across guides -> the data heterogeneity the effect rides on

If the first is comparable to or larger than the second, a single-seed difference
of the size seen in A1 is not distinguishable from seed noise.

Also reports, per guide, whether all three seeds agree on the DIRECTION of the
activity_on - activity_off change, which is a distribution-free stability check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRIC = "mse"
BUDGET = "100"


def load_arm(state_dir: Path, arm_dir_name: str, seed: int) -> pd.Series:
    frames = []
    for path in sorted(state_dir.glob(f"fold_*/{arm_dir_name}/seed{seed}/score/per_condition.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no scores for {arm_dir_name}/seed{seed}")
    frame = pd.concat(frames, ignore_index=True)
    subset = frame[(frame["metric"] == METRIC) & (frame["deg_budget"].astype(str) == BUDGET)]
    return subset.set_index("condition")["value"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Between-seed vs between-guide variance.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--state", default="esc")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    state_dir = args.results_root / "gse283614" / args.state
    args.output_dir.mkdir(parents=True, exist_ok=True)

    on_seeds = {7: load_arm(state_dir, "activity_on", 7)}
    for extra in sorted(state_dir.glob("fold_0/activity_on_seed*")):
        seed = int(extra.name.split("seed")[1])
        on_seeds[seed] = load_arm(state_dir, f"activity_on_seed{seed}", seed)
    off = load_arm(state_dir, "activity_off", 7)

    shared = set(off.index)
    for series in on_seeds.values():
        shared &= set(series.index)
    shared = sorted(shared)

    matrix = pd.DataFrame({seed: series.loc[shared] for seed, series in on_seeds.items()})
    off_vec = off.loc[shared]

    within_guide_between_seed = matrix.std(axis=1, ddof=1)
    between_guide_sd = matrix.mean(axis=1).std(ddof=1)

    deltas = matrix.sub(off_vec, axis=0)
    # Direction agreement across seeds, per guide: all three deltas the same sign.
    signs = np.sign(deltas.to_numpy())
    consistent = np.all(signs == signs[:, [0]], axis=1)
    # Seed-mean delta and its own between-seed spread: the A1 effect, re-derived.
    mean_delta = deltas.mean(axis=1)
    delta_seed_sd = deltas.std(axis=1, ddof=1)

    per_guide = pd.DataFrame(
        {
            "guide": shared,
            "activity_off_mse": off_vec.to_numpy(),
            **{f"activity_on_seed{seed}_mse": matrix[seed].to_numpy() for seed in matrix.columns},
            "mean_delta": mean_delta.to_numpy(),
            "delta_seed_sd": delta_seed_sd.to_numpy(),
            "direction_consistent_across_seeds": consistent,
        }
    )
    per_guide.to_csv(args.output_dir / "seed_variance_per_guide.csv", index=False)

    n_seeds = matrix.shape[1]
    summary = {
        "state": args.state,
        "metric": f"{METRIC}@{BUDGET}",
        "n_seeds": int(n_seeds),
        "seeds": [int(s) for s in matrix.columns],
        "n_guides": len(shared),
        "median_within_guide_between_seed_sd": float(within_guide_between_seed.median()),
        "mean_within_guide_between_seed_sd": float(within_guide_between_seed.mean()),
        "between_guide_sd_of_seed_means": float(between_guide_sd),
        "ratio_seed_over_guide": float(within_guide_between_seed.median() / between_guide_sd),
        "median_mean_delta_activity_on_minus_off": float(mean_delta.median()),
        "mean_delta_activity_on_minus_off": float(mean_delta.mean()),
        "pooled_seed_mean_delta": float(deltas.to_numpy().mean()),
        "n_guides_direction_consistent": int(consistent.sum()),
        "frac_guides_direction_consistent": float(consistent.mean()),
        "n_guides_mean_delta_negative_better_mse": int((mean_delta < 0).sum()),
    }
    # Paired test of the seed-averaged arm against the baseline, for comparison with A1's single seed.
    from scipy import stats

    t = stats.ttest_rel(matrix.mean(axis=1).to_numpy(), off_vec.to_numpy())
    w = stats.wilcoxon(matrix.mean(axis=1).to_numpy(), off_vec.to_numpy())
    summary["paired_t_p_seed_averaged"] = float(t.pvalue)
    summary["wilcoxon_p_seed_averaged"] = float(w.pvalue)

    pd.DataFrame([summary]).to_csv(args.output_dir / "seed_variance_summary.csv", index=False)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
