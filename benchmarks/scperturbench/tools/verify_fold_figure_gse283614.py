"""Independent verification of the ESC mid-band figure numbers.

Recomputes the six metrics straight from the scored per-condition CSVs, with the
GSE132080 paired statistics, so the figure and this check cannot agree by
construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WS = Path(__file__).resolve().parents[1]
RES = WS / ".scperturbench/results/gse283614_activity_ablation_kd3/gse283614/esc"
AS = WS / "reports/assets/gse283614_knockdown3"
FOLD = 1

HIGHER = {"mse": False, "pcc_delta": True, "e_distance": False,
          "wasserstein": False, "kl_divergence": False, "common_degs": True}
PRIMARY = [("mse", "100"), ("pcc_delta", "100"), ("e_distance", "100"),
           ("wasserstein", "100"), ("kl_divergence", "100"), ("common_degs", "cutoff")]

FAIL = []


def check(label, ok, detail=""):
    print(f"[{'ok  ' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def load(arm):
    return pd.read_csv(RES / f"fold_{FOLD}" / arm / "seed7/score/per_condition.csv")


off, on = load("activity_off"), load("activity_on")
conds = sorted(set(off["condition"]) & set(on["condition"]))
check(f"paired guides in fold {FOLD}", len(conds) == 11, f"n={len(conds)}")

# the figure's own table
fig_tests = pd.read_csv(AS / f"fold{FOLD}_esc_activity_on/fold_tests.csv")
fig_pairs = pd.read_csv(AS / f"fold{FOLD}_esc_activity_on/fold_pairwise.csv")
fig_pc = pd.read_csv(AS / f"fold{FOLD}_esc_activity_on/fold_per_condition.csv")

for metric, budget in PRIMARY:
    def vals(df):
        s = df[(df["metric"] == metric) & (df["deg_budget"].astype(str) == budget)]
        return s.set_index("condition")["value"].loc[conds]

    a, b = vals(on).to_numpy(float), vals(off).to_numpy(float)
    diff = a - b
    pct = 100 * diff.mean() / b.mean()
    n = len(diff)
    g = diff.mean() / diff.std(ddof=1) * (1 - 3 / (4 * n - 5))
    p_t = float(stats.ttest_rel(a, b).pvalue)
    p_w = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)

    row_t = fig_tests[(fig_tests["metric"] == metric) & (fig_tests["deg_budget"].astype(str) == budget)].iloc[0]
    row_p = fig_pairs[(fig_pairs["metric"] == metric) & (fig_pairs["deg_budget"].astype(str) == budget)].iloc[0]
    check(f"{metric}@{budget} pct", abs(row_p["relative_change_pct"] - pct) < 1e-6, f"{pct:+.3f}%")
    check(f"{metric}@{budget} t p", abs(row_t["paired_t_p"] - p_t) < 1e-9, f"{p_t:.5f}")
    check(f"{metric}@{budget} wilcoxon p", abs(row_t["wilcoxon_p"] - p_w) < 1e-9, f"{p_w:.5f}")
    check(f"{metric}@{budget} hedges g", abs(row_t["hedges_g_paired"] - g) < 1e-9, f"{g:+.4f}")
    print(f"       off={b.mean():.5g} on={a.mean():.5g}  {'on better' if (pct>0)==HIGHER[metric] else 'on worse'}")

# long-frame integrity: one row per guide per metric per budget
counts = fig_pc.groupby(["metric", "deg_budget"]).size()
check("per-condition rows = 11 guides each", bool((counts == 11).all()), f"unique sizes={sorted(set(counts))}")

# the two low-activity guides should be flagged non-zero / zero as recorded
guides = pd.read_csv(AS / f"fold{FOLD}_esc_activity_on/fold_guides.csv")
nz = int((guides["activity_input"] > 0).sum())
check("guides table reports 8 non-zero activities", nz == 8, f"n_nonzero={nz}")
# This band holds 8 measured and 3 unmeasured guides (NANOG-1-4, NANOG-pro4,
# OCT4-pro2 are absent from the ESC screen).  The figure caption states exactly
# that, so the assertion must match the band, not the whole dataset.
check("band has 8 measured / 3 unmeasured", int(guides["activity_measured"].sum()) == 8
      and int((~guides["activity_measured"]).sum()) == 3,
      f"measured={int(guides['activity_measured'].sum())} unmeasured={int((~guides['activity_measured']).sum())}")

# and the important caveat: activity_on beats mask on the same band
mk = load("activity_mask")
print()
for metric, budget in PRIMARY:
    def mvals(df):
        s = df[(df["metric"] == metric) & (df["deg_budget"].astype(str) == budget)]
        return s.set_index("condition")["value"].loc[conds]
    o, m = mvals(on).to_numpy(float), mvals(mk).to_numpy(float)
    better = (o.mean() > m.mean()) == HIGHER[metric]
    print(f"  on vs mask {metric:16s} on={o.mean():.5g} mask={m.mean():.5g} -> {'on better' if better else 'mask better'}")

print()
if FAIL:
    print(f"FOLD-FIGURE CHECK FAILED: {len(FAIL)} -> {FAIL}")
    sys.exit(1)
print("FOLD-FIGURE CHECK OK")
