"""Sanity-check the paired statistics against an independent implementation.

Recomputes the Wilcoxon signed-rank test, the paired t-test and the BH q-values directly from the
per-condition files (no dependence on the figure module), and prints the per-condition deltas so the
win/loss split behind each p-value is visible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(
    r"C:\Users\Admin\myprojects\perturbench\.scperturbench\results"
    r"\gse132080_activity_ablation\gse132080"
)
ON, OFF = "activity_on_200ep", "activity_off_200ep"
METRICS = [("mse", "100"), ("pcc_delta", "100"), ("e_distance", "100"),
           ("wasserstein", "100"), ("kl_divergence", "100"), ("common_degs", "cutoff")]


def load(arm: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / arm / "seed7" / "score" / "per_condition.csv")


def bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, p[order][i] * n / (i + 1))
        q[order[i]] = running
    return [float(min(max(v, 0.0), 1.0)) for v in q]


def main() -> int:
    on_all, off_all = load(ON), load(OFF)
    rows = []
    for metric, budget in METRICS:
        sel = lambda f: f[(f["metric"] == metric) & (f["deg_budget"].astype(str) == budget)]
        a = sel(on_all).set_index("condition")["value"]
        b = sel(off_all).set_index("condition")["value"]
        joined = pd.concat([a.rename("on"), b.rename("off")], axis=1).dropna()
        diff = (joined["on"] - joined["off"]).to_numpy()
        w_two = stats.wilcoxon(joined["on"], joined["off"], alternative="two-sided")
        w_greater = stats.wilcoxon(joined["on"], joined["off"], alternative="greater")
        w_less = stats.wilcoxon(joined["on"], joined["off"], alternative="less")
        t = stats.ttest_rel(joined["on"], joined["off"])
        rows.append(
            {
                "metric": metric,
                "n": len(joined),
                "n_on_lower": int((diff < 0).sum()),
                "n_on_higher": int((diff > 0).sum()),
                "zeros": int((diff == 0).sum()),
                "wilcoxon_p_two": float(w_two.pvalue),
                "wilcoxon_p_greater": float(w_greater.pvalue),
                "wilcoxon_p_less": float(w_less.pvalue),
                "t_p": float(t.pvalue),
            }
        )
        print(f"=== {metric}@{budget} (n={len(joined)}) ===")
        print(f"  activity lower in {int((diff<0).sum())}, higher in {int((diff>0).sum())}, tied {int((diff==0).sum())}")
        print(joined.assign(delta=diff).sort_values("delta").to_string(float_format=lambda v: f"{v:.5g}"))
        print()
    table = pd.DataFrame(rows)
    table["bh_q_two"] = bh(table["wilcoxon_p_two"].tolist())
    print("=== summary ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
