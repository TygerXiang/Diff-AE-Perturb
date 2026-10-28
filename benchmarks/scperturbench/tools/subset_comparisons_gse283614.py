"""Host-side paired comparison restricted to guides with a strictly positive
activity value.

Answers the question §5.4 of the GSE283614 report needs: does knowing the
**magnitude** of a measured activity beat knowing only **whether** it was measured?

`activity_mask` encodes measured-or-not, but it only differs from `activity_on`
where the value was non-zero (both encode 0 for unmeasured guides, and
`activity_mask` additionally maps the three "measured but exactly 0" guides to
0).  So the informative comparison is `activity_on` vs `activity_mask` on the 22
guides whose value is > 0, plus `activity_on` vs `activity_off` on the same set.

Statistics mirror `scperturbench_ablation_figure` (paired Wilcoxon + paired t +
Hedges g + BH over the six metrics) so the numbers are comparable to the main table.
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
PRIMARY = [
    ("mse", "100"),
    ("pcc_delta", "100"),
    ("e_distance", "100"),
    ("wasserstein", "100"),
    ("kl_divergence", "100"),
    ("common_degs", "cutoff"),
]
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260930


def hedges_g_paired(diff: np.ndarray) -> float:
    n = diff.size
    if n < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diff.mean() / sd * (1 - 3 / (4 * n - 5)))


def bootstrap_paired_delta_ci(on: np.ndarray, off: np.ndarray, rng) -> tuple[float, float]:
    if on.size < 2:
        return float("nan"), float("nan")
    diff = on - off
    idx = rng.integers(0, diff.size, size=(BOOTSTRAP_B, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        q[order[i]] = running
    return [float(min(max(v, 0.0), 1.0)) for v in q]


def load_arm(state_dir: Path, arm_name: str, seed: int) -> pd.DataFrame:
    frames = []
    for path in sorted(state_dir.glob(f"fold_*/{arm_name}/seed{seed}/score/per_condition.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no per-condition scores for {arm_name}/seed{seed}")
    frame = pd.concat(frames, ignore_index=True)
    frame["arm"] = arm_name
    return frame


def compare(
    state_dir: Path,
    conditions: dict,
    arm_on: str,
    arm_off: str,
    subset: str,
    seed: int,
    rng,
) -> pd.DataFrame:
    per_condition = conditions["per_condition"]
    evaluated = list(conditions["evaluated_conditions"])
    if subset == "positive_activity":
        members = [g for g in evaluated if per_condition[g]["activity_input"] > 0]
    elif subset == "measured":
        members = [g for g in evaluated if per_condition[g]["activity_measured"]]
    elif subset == "unmeasured":
        members = [g for g in evaluated if not per_condition[g]["activity_measured"]]
    else:
        members = evaluated

    frame = pd.concat(
        [load_arm(state_dir, arm_off, seed), load_arm(state_dir, arm_on, seed)],
        ignore_index=True,
    )
    frame = frame[frame["condition"].isin(members)]

    rows = []
    for metric, budget in PRIMARY:
        subset_frame = frame[
            (frame["metric"] == metric) & (frame["deg_budget"].astype(str) == budget)
        ]
        wide = subset_frame.pivot_table(
            index="condition", columns="arm", values="value", aggfunc="mean"
        ).dropna(how="any")
        if wide.empty:
            continue
        on = wide[arm_on].to_numpy(dtype=float)
        off = wide[arm_off].to_numpy(dtype=float)
        diff = on - off
        try:
            p_w = float(stats.wilcoxon(on, off, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p_w = float("nan")
        p_t = float(stats.ttest_rel(on, off).pvalue)
        lo, hi = bootstrap_paired_delta_ci(on, off, rng)
        direction = 1 if HIGHER_IS_BETTER[metric] else -1
        rows.append(
            {
                "state": state_dir.name,
                "comparison": f"{arm_on}_vs_{arm_off}",
                "subset": subset,
                "n_conditions": int(on.size),
                "metric": metric,
                "deg_budget": budget,
                "mean_arm_on": float(on.mean()),
                "mean_arm_off": float(off.mean()),
                "mean_paired_delta": float(diff.mean()),
                "paired_delta_ci_low": lo,
                "paired_delta_ci_high": hi,
                "relative_change_pct": float(100 * diff.mean() / off.mean()) if off.mean() else float("nan"),
                "arm_on_better": bool(direction * diff.mean() > 0),
                "n_conditions_arm_on_better": int((direction * diff > 0).sum()),
                "wilcoxon_p": p_w,
                "paired_t_p": p_t,
                "hedges_g_paired": hedges_g_paired(diff),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_q_bh"] = benjamini_hochberg(out["wilcoxon_p"].tolist())
        out["paired_t_q_bh"] = benjamini_hochberg(out["paired_t_p"].tolist())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Subset-restricted paired comparisons.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--state", default="esc")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    conditions = json.loads(
        (args.inputs_dir / f"conditions_{args.state}.json").read_text(encoding="utf-8")
    )
    state_dir = args.results_root / "gse283614" / args.state
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("activity_on", "activity_off", "positive_activity"),
        ("activity_on", "activity_mask", "positive_activity"),
        ("activity_on", "activity_perm", "positive_activity"),
        ("activity_on", "activity_off", "all"),
    ]
    frames = []
    for arm_on, arm_off, subset in comparisons:
        try:
            result = compare(
                state_dir,
                conditions,
                arm_on,
                arm_off,
                subset,
                args.seed,
                np.random.default_rng(BOOTSTRAP_SEED),
            )
        except SystemExit as exc:
            print(f"  skip {arm_on} vs {arm_off} [{subset}]: {exc}")
            continue
        frames.append(result)
    if not frames:
        raise SystemExit("no comparisons produced")
    out = pd.concat(frames, ignore_index=True)
    path = args.output_dir / f"subset_comparisons_{args.state}.csv"
    out.to_csv(path, index=False)

    # The grouped column stores the comparison as "<arm_on>_vs_<arm_off>", so the
    # key is a 2-tuple (comparison, subset); split the arm names back out for the
    # per-comparison headings rather than unpacking the key into three values.
    for (comparison, subset), group in out.groupby(["comparison", "subset"], sort=False):
        arm_on, _, arm_off = comparison.partition("_vs_")
        print(f"\n=== {arm_on} vs {arm_off} on {subset} (n={int(group['n_conditions'].iloc[0])}) ===")
        print(
            group[
                [
                    "metric",
                    "mean_arm_off",
                    "mean_arm_on",
                    "relative_change_pct",
                    "arm_on_better",
                    "wilcoxon_p",
                    "wilcoxon_q_bh",
                    "paired_t_p",
                    "hedges_g_paired",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:.4g}")
        )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
