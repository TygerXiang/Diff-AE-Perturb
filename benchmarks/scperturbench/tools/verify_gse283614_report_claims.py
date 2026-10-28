"""Final consistency check: every headline number in the report must be reproducible
from the CSVs the pipeline wrote.

Exit code 0 means the report's claims and the artifact CSVs agree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

WS = Path(__file__).resolve().parents[1]
RES = WS / ".scperturbench/results/gse283614_activity_ablation/gse283614"
INP = WS / ".scperturbench/inputs/gse283614_activity_ablation"
ASSETS = WS / "reports/assets/gse283614_activity_ablation"

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if condition else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def tests(state: str, analysis: str = "ablation_analysis") -> pd.DataFrame:
    return pd.read_csv(RES / state / analysis / "ablation_tests.csv")


def row(df: pd.DataFrame, metric: str, budget: str) -> pd.Series:
    return df[(df["metric"] == metric) & (df["deg_budget"].astype(str) == budget)].iloc[0]


# ---- ESC main comparison -------------------------------------------------
esc = tests("esc")
check("ESC n_conditions == 33", int(row(esc, "mse", "100")["n_conditions"]) == 33)
r = row(esc, "kl_divergence", "100")
check("ESC kl_divergence +9.8%", abs(r["relative_change_pct"] - 9.823) < 0.01, f"{r['relative_change_pct']:.3f}")
check("ESC kl_divergence q=0.0014", abs(r["wilcoxon_q_bh"] - 0.001446) < 1e-5, f"{r['wilcoxon_q_bh']:.6f}")
check("ESC kl_divergence is the only BH-significant", int(esc["significant_q_lt_0.05"].sum()) == 1,
      f"{int(esc['significant_q_lt_0.05'].sum())} significant")
r = row(esc, "pcc_delta", "100")
check("ESC pcc_delta +36.2%", abs(r["relative_change_pct"] - 36.177) < 0.01, f"{r['relative_change_pct']:.3f}")
check("ESC pcc_delta q=0.058 (not significant)", abs(r["wilcoxon_q_bh"] - 0.058155) < 1e-4, f"{r['wilcoxon_q_bh']:.5f}")
r = row(esc, "mse", "100")
check("ESC mse -3.9%", abs(r["relative_change_pct"] + 3.859) < 0.01, f"{r['relative_change_pct']:.3f}")

# ---- ESC placebo ---------------------------------------------------------
perm = tests("esc", "ablation_analysis_permutation")
pl = perm[perm["deg_budget"].isin(["100", "cutoff"])].copy()
pl["better"] = pl.apply(
    lambda x: (x["mean_paired_delta_on_minus_off"] > 0)
    if x["higher_is_better"]
    else (x["mean_paired_delta_on_minus_off"] < 0),
    axis=1,
)
check("ESC placebo: no primary metric improves", not pl["better"].any(), f"improved={pl['better'].sum()}")

# ---- ESC mask ------------------------------------------------------------
mask = tests("esc", "ablation_analysis_mask")
r = row(mask, "pcc_delta", "100")
check("ESC mask pcc_delta +29.1% q=0.0091", abs(r["relative_change_pct"] - 29.12) < 0.05 and abs(r["wilcoxon_q_bh"] - 0.009089) < 1e-5,
      f"{r['relative_change_pct']:.2f}% q={r['wilcoxon_q_bh']:.6f}")

# ---- ESC subset comparisons ---------------------------------------------
sc = pd.read_csv(ASSETS / "subset_comparisons_esc.csv")
sm = sc[(sc["comparison"] == "activity_on_vs_activity_mask") & (sc["subset"] == "positive_activity")]
check("ESC on-vs-mask positive-activity: none significant", (sm["wilcoxon_q_bh"] >= 0.99).all(),
      f"min q={sm['wilcoxon_q_bh'].min():.4f}")
sp = sc[(sc["comparison"] == "activity_on_vs_activity_perm") & (sc["subset"] == "positive_activity")]
better = sum(
    1 for _, x in sp.iterrows()
    if (x["mean_paired_delta"] > 0) == (x["metric"] in ("pcc_delta", "common_degs"))
)
check("ESC on-vs-perm positive-activity: 6/6 favour activity_on", better == 6, f"{better}/6")

# ---- DE main -------------------------------------------------------------
de = tests("de")
check("DE n_conditions == 31", int(row(de, "mse", "100")["n_conditions"]) == 31)
prim = de[de["deg_budget"].isin(["100", "cutoff"])].copy()
prim["better"] = prim.apply(
    lambda x: (x["mean_paired_delta_on_minus_off"] > 0)
    if x["higher_is_better"]
    else (x["mean_paired_delta_on_minus_off"] < 0),
    axis=1,
)
check("DE: no primary metric improves", not prim["better"].any(), f"improved={prim['better'].sum()}")
r = row(de, "e_distance", "100")
check("DE e_distance +5.7% q=0.0386", abs(r["relative_change_pct"] - 5.698) < 0.01 and abs(r["wilcoxon_q_bh"] - 0.0386) < 1e-3,
      f"{r['relative_change_pct']:.3f}% q={r['wilcoxon_q_bh']:.4f}")

# ---- DE mask (the control that rules out "low coverage") -----------------
dm = tests("de", "ablation_analysis_mask")
dmp = dm[dm["deg_budget"].isin(["100", "cutoff"])].copy()
dmp["better"] = dmp.apply(
    lambda x: (x["mean_paired_delta_on_minus_off"] > 0)
    if x["higher_is_better"]
    else (x["mean_paired_delta_on_minus_off"] < 0),
    axis=1,
)
check("DE mask: also no primary metric improves", not dmp["better"].any(), f"improved={dmp['better'].sum()}")

# ---- probe ---------------------------------------------------------------
probe = pd.read_csv(ASSETS / "probe_all.csv")
for state in ("esc", "de"):
    sub = probe[probe["state"] == state]
    zero = sub[sub["activity_delta"].abs() < 1e-12]
    check(f"{state} probe: identical-value swaps move nothing", bool((zero["l2_shift_swapped"] == 0).all()),
          f"max shift={zero['l2_shift_swapped'].max()}")

# ---- seed variance -------------------------------------------------------
sdf = pd.read_csv(ASSETS / "seed_variance_summary.csv")
check("seed variance: 3 seeds, 33 guides", int(sdf["n_seeds"].iloc[0]) == 3 and int(sdf["n_guides"].iloc[0]) == 33)
check("seed/guide variance ratio < 0.1", float(sdf["ratio_seed_over_guide"].iloc[0]) < 0.1,
      f"{float(sdf['ratio_seed_over_guide'].iloc[0]):.4f}")
check("seed-averaged mse effect still null", float(sdf["paired_t_p_seed_averaged"].iloc[0]) > 0.5,
      f"p={float(sdf['paired_t_p_seed_averaged'].iloc[0]):.4f}")

# ---- coverage claims -----------------------------------------------------
diag = json.loads((WS / ".scperturbench/inputs/gse283614_activity_diagnostic.json").read_text(encoding="utf-8"))
check("ESC coverage 25/33", diag["states"]["esc"]["n_measured"] == 25 and diag["states"]["esc"]["n_evaluated"] == 33)
check("DE coverage 19/31", diag["states"]["de"]["n_measured"] == 19 and diag["states"]["de"]["n_evaluated"] == 31)
check("ESC 3 measured zeros", diag["states"]["esc"]["activity_distribution"]["n_zero"] == 3)
check("DE 10 measured zeros", diag["states"]["de"]["activity_distribution"]["n_zero"] == 10)

# ---- proxy quality -------------------------------------------------------
ctx = json.loads((ASSETS / "activity_context.json").read_text(encoding="utf-8"))
pq = ctx["proxy_quality"]
check("ESC proxy overall rho=0.575 p=0.0027",
      abs(pq["overall"]["spearman_rho"] - 0.5746) < 1e-3 and abs(pq["overall"]["p_value"] - 0.00266) < 1e-5)
check("ESC proxy within OCT4 rho=0.789", abs(pq["within_target"]["OCT4"]["spearman_rho"] - 0.7885) < 1e-3)
check("ESC proxy within NANOG rho=0.041", abs(pq["within_target"]["NANOG"]["spearman_rho"] - 0.0410) < 1e-3)
check("preflight all passed", json.loads((WS / ".scperturbench/inputs/gse283614_activity_preflight.json").read_text(encoding="utf-8"))["all_passed"])

print()
if FAILURES:
    print(f"REPORT CONSISTENCY CHECK FAILED: {len(FAILURES)} -> {FAILURES}")
    sys.exit(1)
print("REPORT CONSISTENCY CHECK OK")
