"""Verify every number quoted in the kd3 report against the artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WS = Path(__file__).resolve().parents[1]
KD = WS / ".scperturbench/results/gse283614_activity_ablation_kd3/gse283614"
AS = WS / "reports/assets/gse283614_knockdown3"

FAIL: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def tests(state: str, analysis: str = "ablation_analysis") -> pd.DataFrame:
    return pd.read_csv(KD / state / analysis / "ablation_tests.csv")


def row(df: pd.DataFrame, metric: str, budget: str) -> pd.Series:
    return df[(df["metric"] == metric) & (df["deg_budget"].astype(str) == budget)].iloc[0]


# ---- kd3 ESC main --------------------------------------------------------
esc = tests("esc")
check("kd3 ESC n=33", int(row(esc, "mse", "100")["n_conditions"]) == 33)
r = row(esc, "kl_divergence", "100")
check("kl +11.3% q=7.2e-05", abs(r["relative_change_pct"] - 11.3) < 0.05 and abs(r["wilcoxon_q_bh"] - 7.169e-05) < 1e-7,
      f"{r['relative_change_pct']:.2f}% q={r['wilcoxon_q_bh']:.3g}")
r = row(esc, "pcc_delta", "100")
check("pcc_delta +14.2% not significant", abs(r["relative_change_pct"] - 14.17) < 0.05 and r["wilcoxon_q_bh"] > 0.2,
      f"{r['relative_change_pct']:.2f}% q={r['wilcoxon_q_bh']:.3f}")
check("mse -9.6%", abs(row(esc, "mse", "100")["relative_change_pct"] + 9.555) < 0.05)
check("common_degs +14.1%", abs(row(esc, "common_degs", "cutoff")["relative_change_pct"] - 14.14) < 0.05)

# ---- mask reversal -------------------------------------------------------
mask = tests("esc", "ablation_analysis_mask")
r = row(mask, "mse", "100")
check("kd3 mask mse is WORSE (+2.1%)", r["relative_change_pct"] > 0, f"{r['relative_change_pct']:+.2f}%")
r = row(mask, "pcc_delta", "100")
check("kd3 mask pcc_delta is WORSE (-24%)", r["relative_change_pct"] < 0, f"{r['relative_change_pct']:+.2f}%")
r = row(mask, "e_distance", "100")
check("kd3 mask e_distance worse, t p=0.039", r["relative_change_pct"] > 0 and abs(r["paired_t_p"] - 0.03868) < 1e-4)

# activity_on must beat mask on 5/6 primary metrics
on = tests("esc")
prim = [("mse", "100", False), ("pcc_delta", "100", True), ("e_distance", "100", False),
        ("wasserstein", "100", False), ("kl_divergence", "100", False), ("common_degs", "cutoff", True)]
better = 0
for metric, budget, higher in prim:
    a = row(on, metric, budget)["mean_activity_on"]
    m = row(mask, metric, budget)["mean_activity_on"]
    if (a > m) == higher:
        better += 1
check("activity_on beats mask on 5/6 metrics", better == 5, f"{better}/6")

# ---- placebo -------------------------------------------------------------
perm = tests("esc", "ablation_analysis_permutation")
r = row(perm, "e_distance", "100")
check("placebo e_distance +20.8% q=0.013", abs(r["relative_change_pct"] - 20.82) < 0.05 and abs(r["wilcoxon_q_bh"] - 0.01341) < 1e-4,
      f"{r['relative_change_pct']:.2f}% q={r['wilcoxon_q_bh']:.4f}")
r = row(perm, "kl_divergence", "100")
check("placebo kl +11.7% q=8e-05", abs(r["relative_change_pct"] - 11.71) < 0.05)

# ---- DE ------------------------------------------------------------------
de = tests("de")
check("kd3 DE n=31", int(row(de, "mse", "100")["n_conditions"]) == 31)
r = row(de, "common_degs", "cutoff")
check("DE common_degs -16.3% q=0.0031", abs(r["relative_change_pct"] + 16.26) < 0.05 and abs(r["wilcoxon_q_bh"] - 0.003146) < 1e-5,
      f"{r['relative_change_pct']:.2f}% q={r['wilcoxon_q_bh']:.4f}")
r = row(de, "e_distance", "100")
check("DE e_distance +6.1% t p=0.023", abs(r["relative_change_pct"] - 6.077) < 0.05 and abs(r["paired_t_p"] - 0.02329) < 1e-4)
dm = tests("de", "ablation_analysis_mask")
r = row(dm, "common_degs", "cutoff")
check("DE mask common_degs -18.4% q=7e-05", abs(r["relative_change_pct"] + 18.39) < 0.05 and abs(r["wilcoxon_q_bh"] - 7.006e-05) < 1e-7)

# ---- per-band table quoted in the report ---------------------------------
pb = pd.read_csv(AS / "per_band_esc.csv")
on_band = pb[pb["arm"] == "activity_on"]
def band(fold: int, metric: str) -> float:
    s = on_band[(on_band["fold"] == fold) & (on_band["metric"] == metric)]
    return float(s["relative_change_pct"].iloc[0])
check("band fold0 low pcc_delta -35.8%", abs(band(0, "pcc_delta") + 35.8) < 0.2, f"{band(0,'pcc_delta'):+.1f}%")
check("band fold1 mid mse -13.0%", abs(band(1, "mse") + 13.0) < 0.2, f"{band(1,'mse'):+.1f}%")
check("band fold1 mid pcc_delta +327.7%", abs(band(1, "pcc_delta") - 327.7) < 1.0, f"{band(1,'pcc_delta'):+.1f}%")
check("band fold2 high mse -11.7%", abs(band(2, "mse") + 11.7) < 0.2, f"{band(2,'mse'):+.1f}%")
check("band best band is mid, not high", abs(band(1, "mse")) > abs(band(2, "mse")) and band(2, "mse") < 0)

# ---- probe ---------------------------------------------------------------
pr = pd.read_csv(AS / "probe_summary.csv")
e = pr[pr["state"] == "esc"].iloc[0]
check("kd3 ESC probe zero-delta shift is 0", float(e["max_shift_on_zero_delta_swap"]) == 0.0)
check("kd3 ESC shift/floor 0.933", abs(float(e["median_shift_over_noise_floor"]) - 0.933) < 0.01)

print()
if FAIL:
    print(f"KD3 REPORT CHECK FAILED: {len(FAIL)} -> {FAIL}")
    sys.exit(1)
print("KD3 REPORT CHECK OK")
