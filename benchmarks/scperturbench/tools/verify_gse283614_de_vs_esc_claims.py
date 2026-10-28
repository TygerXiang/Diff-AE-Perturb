"""Verify every number quoted in reports/gse283614_de_vs_esc_why.md.

Exits 1 on any mismatch.  Only reads artifacts under reports/assets/gse283614_de_vs_esc/
plus the two published result trees (read-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parents[1]
A = WS / "reports/assets/gse283614_de_vs_esc"
KD3 = WS / ".scperturbench/results/gse283614_activity_ablation_kd3/gse283614"
CV = WS / ".scperturbench/results/gse283614_activity_ablation/gse283614"
PRIMARY = ["mse@100", "pcc_delta@100", "e_distance@100",
           "wasserstein@100", "kl_divergence@100", "common_degs@cutoff"]

FAIL: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def close(a, b, tol=0.02) -> bool:
    return bool(np.isfinite(a) and abs(float(a) - float(b)) <= tol)


def keyed(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["key"] = frame["metric"] + "@" + frame["deg_budget"].astype(str)
    return frame


# ---------------------------------------------------------------- summary.json
summary = json.loads((A / "summary.json").read_text(encoding="utf-8"))
primary = summary["primary_effect"]
check("summary has kd3 DE common_degs", close(primary.get("kd3_de_common_degs_pct"), -16.26))
check("summary has dose_cv DE e_distance", close(primary.get("dose_cv_de_e_distance_pct"), 5.698))

# ------------------------------------------------------------ state_parallel
sp = keyed(pd.read_csv(A / "state_parallel.csv"))
prim = sp[sp["key"].isin(PRIMARY)]


def cell(design: str, state: str, key: str) -> pd.Series:
    return prim[(prim["design"] == design) & (prim["state"] == state) & (prim["key"] == key)].iloc[0]


check("kd3 DE n=31 / ESC n=33",
      int(cell("kd3", "de", "mse@100")["n_conditions"]) == 31
      and int(cell("kd3", "esc", "mse@100")["n_conditions"]) == 33)
check("kd3 DE mse +24.03%", close(cell("kd3", "de", "mse@100")["relative_change_pct"], 24.03))
check("kd3 DE pcc_delta -19.99%", close(cell("kd3", "de", "pcc_delta@100")["relative_change_pct"], -19.99))
check("kd3 DE e_distance +6.077%", close(cell("kd3", "de", "e_distance@100")["relative_change_pct"], 6.077))
check("kd3 DE wasserstein +2.108%", close(cell("kd3", "de", "wasserstein@100")["relative_change_pct"], 2.108))
check("kd3 DE kl +2.033%", close(cell("kd3", "de", "kl_divergence@100")["relative_change_pct"], 2.033))
r = cell("kd3", "de", "common_degs@cutoff")
check("kd3 DE common_degs -16.26% q=0.0031",
      close(r["relative_change_pct"], -16.26) and close(r["q_bh"], 0.003146, 1e-6),
      f"{r['relative_change_pct']:.2f}% q={r['q_bh']:.4g}")
check("kd3 DE is 6/6 worse", not cell("kd3", "de", "mse@100")["better"]
      and not cell("kd3", "de", "pcc_delta@100")["better"]
      and not cell("kd3", "de", "e_distance@100")["better"]
      and not cell("kd3", "de", "wasserstein@100")["better"]
      and not cell("kd3", "de", "kl_divergence@100")["better"]
      and not cell("kd3", "de", "common_degs@cutoff")["better"])
check("kd3 ESC mse -9.555%", close(cell("kd3", "esc", "mse@100")["relative_change_pct"], -9.555))
check("kd3 ESC pcc_delta +14.17%", close(cell("kd3", "esc", "pcc_delta@100")["relative_change_pct"], 14.17))
r = cell("kd3", "esc", "kl_divergence@100")
check("kd3 ESC kl +11.30% q=7.169e-05",
      close(r["relative_change_pct"], 11.30) and close(r["q_bh"], 7.169e-05, 1e-8),
      f"{r['relative_change_pct']:.2f}% q={r['q_bh']:.4g}")
check("kd3 ESC is 4/6 better (e_distance, kl worse)",
      sum(bool(cell("kd3", "esc", k)["better"]) for k in PRIMARY) == 4)
check("kd3 DE is 0/6 better / dose_cv DE is 0/6 better",
      sum(bool(cell("kd3", "de", k)["better"]) for k in PRIMARY) == 0
      and sum(bool(cell("dose_cv", "de", k)["better"]) for k in PRIMARY) == 0)
check("dose_cv ESC is 3/6 better",
      sum(bool(cell("dose_cv", "esc", k)["better"]) for k in PRIMARY) == 3)
check("kd3 DE mse q is NOT significant (direction only)",
      float(cell("kd3", "de", "mse@100")["q_bh"]) > 0.4,
      f"q={cell('kd3', 'de', 'mse@100')['q_bh']:.3f}")

# absolute baselines quoted in section 1.1
check("DE off absolutes (mse/pcc/common_degs)",
      close(cell("kd3", "de", "mse@100")["mean_off"], 0.02958, 1e-4)
      and close(cell("kd3", "de", "pcc_delta@100")["mean_off"], 0.3244, 1e-3)
      and close(cell("kd3", "de", "common_degs@cutoff")["mean_off"], 0.09153, 1e-4))
check("ESC off absolutes (mse/pcc/common_degs)",
      close(cell("kd3", "esc", "mse@100")["mean_off"], 0.07128, 1e-4)
      and close(cell("kd3", "esc", "pcc_delta@100")["mean_off"], 0.2092, 1e-3)
      and close(cell("kd3", "esc", "common_degs@cutoff")["mean_off"], 0.07185, 1e-4))


def de_better_than_esc(key: str) -> bool:
    d, e = cell("kd3", "de", key), cell("kd3", "esc", key)
    lower_is_better = not bool(d["higher_is_better"])
    return bool(d["mean_off"] < e["mean_off"]) if lower_is_better else bool(d["mean_off"] > e["mean_off"])


check("DE baseline better than ESC on 5/6 absolute metrics (wasserstein the exception)",
      sum(de_better_than_esc(k) for k in PRIMARY) == 5,
      f"{sum(de_better_than_esc(k) for k in PRIMARY)}/6")

# dose_cv
check("dose_cv DE mse +24.51%", close(cell("dose_cv", "de", "mse@100")["relative_change_pct"], 24.51))
r = cell("dose_cv", "de", "e_distance@100")
check("dose_cv DE e_distance +5.698% q=0.0386",
      close(r["relative_change_pct"], 5.698) and close(r["q_bh"], 0.0386, 1e-3))
check("dose_cv ESC pcc_delta +36.18%", close(cell("dose_cv", "esc", "pcc_delta@100")["relative_change_pct"], 36.18))
check("dose_cv ESC kl +9.823%", close(cell("dose_cv", "esc", "kl_divergence@100")["relative_change_pct"], 9.823))
check("dose_cv DE common_degs is fragile (-0.57%)",
      close(cell("dose_cv", "de", "common_degs@cutoff")["relative_change_pct"], -0.5681, 1e-3))

# ------------------------------------------------------------- design_compare
dc = pd.read_csv(A / "design_compare.csv")
check("15/20 metric-state cells agree in sign across designs",
      int(dc["sign_agree"].sum()) == 15, f"{int(dc['sign_agree'].sum())}/20")

# ---------------------------------------------------------------- deg overlap
deg = pd.read_csv(A / "deg_overlap.csv")
check("ESC/DE DEG overlap ~17.5 genes, Jaccard ~0.10",
      close(deg["n_overlap"].mean(), 17.545, 0.01) and close(deg["jaccard"].mean(), 0.100, 0.002),
      f"{deg['n_overlap'].mean():.3f} genes, J={deg['jaccard'].mean():.3f}")

# ------------------------------------------------------------------ jackknife
jk = keyed(pd.read_csv(A / "jackknife_folds.csv"))
de_jk = jk[(jk["design"] == "kd3") & (jk["state"] == "de")]


def variant(key: str, name: str) -> pd.Series:
    return de_jk[(de_jk["key"] == key) & (de_jk["variant"] == name)].iloc[0]


check("drop_fold_2 flips mse to -20.64%", close(variant("mse@100", "drop_fold_2")["relative_change_pct"], -20.64))
check("drop_fold_2 flips pcc_delta to +38.94%", close(variant("pcc_delta@100", "drop_fold_2")["relative_change_pct"], 38.94))
check("drop_fold_2 flips e_distance to -1.97%", close(variant("e_distance@100", "drop_fold_2")["relative_change_pct"], -1.97))
check("common_degs stays negative in all four fold variants",
      bool((de_jk[de_jk["key"] == "common_degs@cutoff"]["relative_change_pct"] < 0).all()))

# --------------------------------------------------------------- common guides
cg = keyed(pd.read_csv(A / "common_guides.csv"))
check("31 shared guides in both states, both designs",
      set(cg["n_shared_guides"].unique()) == {31})
cg_de = cg[(cg["design"] == "kd3") & (cg["state"] == "de")]
check("DE still 6/6 worse on the 31 shared guides",
      not bool(cg_de[cg_de["key"].isin(PRIMARY)]["better"].any()))

# -------------------------------------------------------------- bands / targets
tb = keyed(pd.read_csv(A / "per_target_bands.csv"))


def band(group_type: str, group: str, design: str, state: str, key: str) -> pd.Series:
    return tb[(tb["group_type"] == group_type) & (tb["group"] == group) & (tb["design"] == design)
              & (tb["state"] == state) & (tb["key"] == key)].iloc[0]


check("H6 kd3 DE common_degs: NANOG -17.6% / OCT4 -12.6%",
      close(band("target", "NANOG", "kd3", "de", "common_degs@cutoff")["relative_change_pct"], -17.6, 0.06)
      and close(band("target", "POU5F1", "kd3", "de", "common_degs@cutoff")["relative_change_pct"], -12.6, 0.06))
check("H2 kd3 DE measured_nonzero common_degs -8.6%",
      close(band("activity_band", "measured_nonzero", "kd3", "de", "common_degs@cutoff")["relative_change_pct_robust"], -8.6, 0.06))
check("H2 kd3 ESC measured_nonzero: common_degs +38.6%, mse -30.7%",
      close(band("activity_band", "measured_nonzero", "kd3", "esc", "common_degs@cutoff")["relative_change_pct_robust"], 38.6, 0.06)
      and close(band("activity_band", "measured_nonzero", "kd3", "esc", "mse@100")["relative_change_pct_robust"], -30.7, 0.06))
check("H2 kd3 ESC unmeasured mse +46.2%",
      close(band("activity_band", "unmeasured", "kd3", "esc", "mse@100")["relative_change_pct_robust"], 46.2, 0.06))
check("H2 kd3 DE mse/pcc by band (+52.2/-25.4 measured, +16.2/-24.9 zero, +12.4/-26.9 unmeasured)",
      close(band("activity_band", "measured_nonzero", "kd3", "de", "mse@100")["relative_change_pct_robust"], 52.2, 0.06)
      and close(band("activity_band", "measured_zero", "kd3", "de", "mse@100")["relative_change_pct_robust"], 16.2, 0.06)
      and close(band("activity_band", "unmeasured", "kd3", "de", "mse@100")["relative_change_pct_robust"], 12.4, 0.06)
      and close(band("activity_band", "measured_nonzero", "kd3", "de", "pcc_delta@100")["relative_change_pct_robust"], -25.4, 0.06)
      and close(band("activity_band", "unmeasured", "kd3", "de", "pcc_delta@100")["relative_change_pct_robust"], -26.9, 0.06))
check("H2 kd3 ESC zero/unmeasured common_degs -44.5/-31.4",
      close(band("activity_band", "measured_zero", "kd3", "esc", "common_degs@cutoff")["relative_change_pct_robust"], -44.5, 0.06)
      and close(band("activity_band", "unmeasured", "kd3", "esc", "common_degs@cutoff")["relative_change_pct_robust"], -31.4, 0.06))
check("H2 band sizes: DE 9/10/12, ESC 22/3/8",
      [int(band("activity_band", g, "kd3", "de", "mse@100")["n_conditions"])
       for g in ["measured_nonzero", "measured_zero", "unmeasured"]] == [9, 10, 12]
      and [int(band("activity_band", g, "kd3", "esc", "mse@100")["n_conditions"])
           for g in ["measured_nonzero", "measured_zero", "unmeasured"]] == [22, 3, 8])
db = tb[(tb["group_type"] == "dose_band") & (tb["design"] == "kd3") & (tb["state"] == "de")
        & (tb["key"].isin(PRIMARY))]
high = db[db["group"] == "high"]
high_better = sum(bool(v) for v in high["better"])
contains = high["relative_change_pct"].isin(
    [v for v in high["relative_change_pct"]])  # keep series alive for readability
check("H3(c) kd3 DE high-dose tertile: 0/6 better (all six worse)",
      high_better == 0 and len(high) == 6, f"{6 - high_better}/6 worse")
low = db[db["group"] == "low"]
check("H3(c) kd3 DE low-dose tertile is NOT uniformly worse",
      sum(bool(v) for v in low["better"]) >= 2,
      f"{sum(bool(v) for v in low['better'])}/6 better in the low band")

# ----------------------------------------------------------- selective pressure
sc = pd.read_csv(A / "selective_pressure_corr.csv")
scc = pd.read_csv(A / "stratum_composition_corr.csv")


def corr(design: str, state: str, x: str, y: str) -> pd.Series:
    return sc[(sc["design"] == design) & (sc["state"] == state) & (sc["x"] == x) & (sc["y"] == y)].iloc[0]


def dcorr(design: str, state: str, x: str, key: str) -> pd.Series:
    y = f"damage::{key}"
    return scc[(scc["design"] == design) & (scc["state"] == state) & (scc["x"] == x)
               & (scc["y"] == y)].iloc[0]


check("H5 dose vs cell count: DE -0.645 (q=3.6e-4) > ESC -0.464 (q=0.022)",
      close(corr("kd3", "de", "knockdown_fraction", "n_gt_cells")["spearman_rho"], -0.645, 0.002)
      and close(corr("kd3", "de", "knockdown_fraction", "n_gt_cells")["q_bh"], 3.55e-4, 1e-6)
      and close(corr("kd3", "esc", "knockdown_fraction", "n_gt_cells")["spearman_rho"], -0.464, 0.002))
for metric, rho, q in [("mse@100", 0.668, 8.649e-4), ("pcc_delta@100", 0.705, 4.561e-4),
                       ("e_distance@100", 0.561, 6.834e-3), ("wasserstein@100", 0.659, 8.649e-4),
                       ("common_degs@cutoff", 0.621, 2.24e-3), ("kl_divergence@100", 0.205, 0.548)]:
    r = dcorr("kd3", "de", "knockdown_fraction", metric)
    check(f"H5 DE dose vs damage {metric} rho={rho} q={q:g}",
          close(r["spearman_rho"], rho, 0.002) and close(r["q_bh"], q, max(1e-4, q * 0.02)),
          f"rho={r['spearman_rho']:+.3f} q={r['q_bh']:.3g}")
r = dcorr("kd3", "esc", "knockdown_fraction", "common_degs@cutoff")
check("H5 ESC dose vs common_degs damage is negative (p=0.0049, q=0.058)",
      close(r["spearman_rho"], -0.478, 0.002) and close(r["p_value"], 0.004938, 1e-5)
      and close(r["q_bh"], 0.05802, 1e-4), f"rho={r['spearman_rho']:+.3f} p={r['p_value']:.4g} q={r['q_bh']:.3g}")
check("H5 ESC other dose-damage correlations are n.s.",
      all(dcorr("kd3", "esc", "knockdown_fraction", m)["q_bh"] > 0.2
          for m in ["mse@100", "e_distance@100", "wasserstein@100", "kl_divergence@100"]))

# -------------------------------------------------------------- curve fits (H3a)
cf = pd.read_csv(A / "curve_fits_de.csv")


def fit(response: str) -> pd.Series:
    return cf[cf["response"] == response].iloc[0]


check("H3(a) ESC_cluster1_vs_NANOG = linear (paper+repo agree)",
      fit("ESC_cluster1_score_vs_NANOG")["selected_model"] == "linear"
      and bool(fit("ESC_cluster1_score_vs_NANOG")["agrees_with_paper"]))
check("H3(a) ESC_cluster2_vs_OCT4 = hill, ed50 0.513",
      fit("ESC_cluster2_score_vs_OCT4")["selected_model"] == "hill"
      and close(fit("ESC_cluster2_score_vs_OCT4")["ed50_tf_remaining"], 0.5133, 0.001))
check("H3(a) DE differentiation = quadratic inverted-U with interior vertex",
      fit("DE_differentiation_score_vs_NANOG")["selected_model"] == "quadratic"
      and fit("DE_differentiation_score_vs_OCT4")["selected_model"] == "quadratic"
      and bool(fit("DE_differentiation_score_vs_NANOG")["inverted_u"])
      and bool(fit("DE_differentiation_score_vs_OCT4")["inverted_u"]))
check("H3(a) vertices 0.352 / 0.619 (tf_remaining)",
      close(fit("DE_differentiation_score_vs_NANOG")["vertex_tf_remaining"], 0.3521, 0.001)
      and close(fit("DE_differentiation_score_vs_OCT4")["vertex_tf_remaining"], 0.6192, 0.001))
check("H3(a) delta AIC 3.36 / 2.69",
      close(fit("DE_differentiation_score_vs_NANOG")["delta_aic_linear_minus_quadratic"], 3.364, 0.01)
      and close(fit("DE_differentiation_score_vs_OCT4")["delta_aic_linear_minus_quadratic"], 2.692, 0.01))
check("DE off-target vs NANOG = hill",
      fit("DE_off-target_score_vs_NANOG")["selected_model"] == "hill")
check("repo cross-check: 5/6 responses agree on the model family",
      int(cf["agrees_with_repo_fit"].sum()) == 5, f"{int(cf['agrees_with_repo_fit'].sum())}/6")

# -------------------------------------------------------- heterogeneity (H4)
hd = pd.read_csv(A / "heterogeneity_decomposition.csv")
de = hd[(hd["state"] == "de") & (hd["design"] == "kd3")]
esc = hd[(hd["state"] == "esc") & (hd["design"] == "kd3")]
check("H4 kd3 DE between-guide variance share 1.1-2.1%",
      close(100 * de["between_guide_share"].min(), 1.1, 0.05)
      and close(100 * de["between_guide_share"].max(), 2.1, 0.05))
check("H4 kd3 ESC between-guide variance share 2.4-6.3%",
      close(100 * esc["between_guide_share"].min(), 2.4, 0.05)
      and close(100 * esc["between_guide_share"].max(), 6.3, 0.05))
check("H4 kd3 stratum share 3.2-4.1% (DE) / 3.8-4.5% (ESC) - far below the 50% threshold",
      close(100 * de["within_guide_between_stratum_share"].min(), 3.2, 0.05)
      and close(100 * de["within_guide_between_stratum_share"].max(), 4.1, 0.05)
      and close(100 * esc["within_guide_between_stratum_share"].min(), 3.8, 0.05)
      and close(100 * esc["within_guide_between_stratum_share"].max(), 4.5, 0.05))
check("H4 dose_cv stratum share 3.7-4.3% (DE) / 5.0-5.6% (ESC)",
      close(100 * hd[(hd.state == "de") & (hd.design == "dose_cv")]["within_guide_between_stratum_share"].min(), 3.7, 0.05)
      and close(100 * hd[(hd.state == "esc") & (hd.design == "dose_cv")]["within_guide_between_stratum_share"].max(), 5.6, 0.05))

scc = pd.read_csv(A / "stratum_composition_corr.csv")


def comp(state: str, x: str, y: str) -> pd.Series:
    return scc[(scc["design"] == "kd3") & (scc["state"] == state) & (scc["x"] == x) & (scc["y"] == y)].iloc[0]


check("H4 dose vs high-stratum fraction: DE +0.581 / ESC -0.720",
      close(comp("de", "knockdown_fraction", "frac_high")["spearman_rho"], 0.581, 0.002)
      and close(comp("esc", "knockdown_fraction", "frac_high")["spearman_rho"], -0.720, 0.002))
check("H4 composition-correlation p-values: DE 4.0e-4 / ESC 2.3e-6",
      close(comp("de", "knockdown_fraction", "frac_high")["p_value"], 3.967e-4, 1e-6)
      and close(comp("esc", "knockdown_fraction", "frac_high")["p_value"], 2.304e-6, 1e-9))
check("H6 OCT4 ESC pcc_delta has a degenerate raw denominator (-221%/-296%, robust -49.5%/+4.9%)",
      close(band("target", "POU5F1", "kd3", "esc", "pcc_delta@100")["mean_off"], -0.08471, 1e-5)
      and close(band("target", "POU5F1", "dose_cv", "esc", "pcc_delta@100")["mean_off"], -0.08164, 1e-5)
      and close(band("target", "POU5F1", "kd3", "esc", "pcc_delta@100")["relative_change_pct_robust"], -49.49, 0.02)
      and close(band("target", "POU5F1", "dose_cv", "esc", "pcc_delta@100")["relative_change_pct_robust"], 4.869, 0.02))

# ------------------------------------------------------------------- probes (H7)
pc = pd.read_csv(A / "probe_compare.csv")


def probe(design: str, state: str) -> pd.Series:
    return pc[(pc["design"] == design) & (pc["state"] == state)].iloc[0]


check("T7.1 identical-value swaps never move the prediction (max shift 0, all arms)",
      bool((pc["max_shift_of_identical_swaps"].fillna(0) == 0).all()))
check("T7.1 ESC n=8 ratio 0.965 / DE n=8 rerun 1.427",
      close(probe("kd3", "esc")["ratio_shift_over_noise"], 0.9649, 0.001)
      and close(probe("kd3_de_n8", "de")["ratio_shift_over_noise"], 1.427, 0.001))
check("T7.1 hit-zero displacement/noise: DE(n=8) 1.158 < 2.0 and ESC 2.637",
      close(probe("kd3_de_n8", "de")["hit_zero_ratio"], 1.158, 0.002)
      and close(probe("kd3", "esc")["hit_zero_ratio"], 2.637, 0.002))
check("T7.1 DE n=8 exceeds noise floor for 10/14 changed swaps",
      int(probe("kd3_de_n8", "de")["n_exceed_noise_floor"]) == 10
      and int(probe("kd3_de_n8", "de")["n_changed"]) == 14)
check("T7.1 n_samples differ as reported (ESC 8, DE published 4, DE rerun 8)",
      int(probe("kd3", "esc")["n_samples"]) == 8 and int(probe("kd3", "de")["n_samples"]) == 4
      and int(probe("kd3_de_n8", "de")["n_samples"]) == 8)

pv = keyed(pd.read_csv(A / "probe_vs_harm.csv"))


def harm(probe_design: str, state: str, key: str, scope: str) -> pd.Series:
    return pv[(pv["probe_design"] == probe_design) & (pv["state"] == state) & (pv["key"] == key)
              & (pv["scope"] == scope) & (pv["ratio_column"] == "ratio")].iloc[0]


RHO = "spearman_rho_ratio_vs_damage"
check("T7.2 DE e_distance pooled +0.480 (p=0.037) but +0.144 (p=0.66) among measured guides",
      close(harm("kd3_de_n8", "de", "e_distance@100", "all_changed_swaps")[RHO], 0.480, 0.002)
      and close(harm("kd3_de_n8", "de", "e_distance@100", "measured")[RHO], 0.144, 0.002)
      and harm("kd3_de_n8", "de", "e_distance@100", "measured")["p_value"] > 0.5)
check("T7.2 DE e_distance is strongest among UNMEASURED guides (+0.757, p=0.049)",
      close(harm("kd3_de_n8", "de", "e_distance@100", "unmeasured")[RHO], 0.757, 0.002)
      and harm("kd3_de_n8", "de", "e_distance@100", "unmeasured")["p_value"] < 0.05)
check("T7.2 ESC shift correlates NEGATIVELY with damage (pcc_delta -0.573, measured -0.450)",
      close(harm("kd3", "esc", "pcc_delta@100", "all_changed_swaps")[RHO], -0.573, 0.002)
      and close(harm("kd3", "esc", "pcc_delta@100", "measured")[RHO], -0.450, 0.002))

# -------------------------------------------------------- response shape (H3b)
rs = pd.read_csv(A / "response_shape_summary.csv")


def shape(design: str, state: str, arm: str) -> pd.Series:
    return rs[(rs["design"] == design) & (rs["state"] == state) & (rs["arm"] == arm)].iloc[0]


check("H3(b) kd3 DE truth non-monotone 74.65% / on 77.76% (gap -3.1pt, not -30)",
      close(shape("kd3", "de", "activity_on")["share_nonmonotone_pct_truth"], 74.646, 0.01)
      and close(shape("kd3", "de", "activity_on")["share_nonmonotone_pct_prediction"], 77.755, 0.01))
check("H3(b) no state reaches the 30-point gap",
      not bool(rs["h3b_truth_exceeds_prediction_by_30pts"].any()))
check("H3(b) DE on-arm inverted-U share rises vs off (47.8% vs 35.6%)",
      close(shape("kd3", "de", "activity_on")["share_inverted_u_prediction"], 0.478, 0.002)
      and close(shape("kd3", "de", "activity_off")["share_inverted_u_prediction"], 0.356, 0.002))
check("H3(b) ESC on-arm inverted-U share falls vs off (28.9% vs 29.5%)",
      close(shape("kd3", "esc", "activity_on")["share_inverted_u_prediction"], 0.289, 0.002)
      and close(shape("kd3", "esc", "activity_off")["share_inverted_u_prediction"], 0.295, 0.002))

rsd = pd.read_csv(A / "response_shape_summary_deg.csv")


def shaped(design: str, state: str, arm: str, source: str) -> pd.Series:
    return rsd[(rsd["design"] == design) & (rsd["state"] == state) & (rsd["arm"] == arm)
               & (rsd["source"] == source)].iloc[0]


check("H3(b) DEG-restricted gene counts: DE 3082 / ESC 2052",
      int(shaped("kd3", "de", "activity_on", "truth")["n_deg_genes"]) == 3082
      and int(shaped("kd3", "esc", "activity_on", "truth")["n_deg_genes"]) == 2052)
check("H3(b) DEG-restricted: DE truth 75.9%/33.6%, on-pred 76.7%/52.1% (+17.5pt inverted-U)",
      close(shaped("kd3", "de", "activity_on", "truth")["share_nonmonotone_pct"], 75.892, 0.01)
      and close(shaped("kd3", "de", "activity_on", "prediction")["share_nonmonotone_pct"], 76.671, 0.01)
      and close(shaped("kd3", "de", "activity_on", "prediction")["share_inverted_u_pct"], 52.109, 0.01)
      and close(shaped("kd3", "de", "activity_off", "prediction")["share_inverted_u_pct"], 34.653, 0.01))
check("H3(b) DEG-restricted: ESC on-pred 61.1%/26.5% (gap +10.3pt, inverted-U -2.8pt)",
      close(shaped("kd3", "esc", "activity_on", "prediction")["share_nonmonotone_pct"], 61.062, 0.01)
      and close(shaped("kd3", "esc", "activity_on", "prediction")["share_inverted_u_pct"], 26.462, 0.01)
      and close(shaped("kd3", "esc", "activity_off", "prediction")["share_inverted_u_pct"], 29.240, 0.01))

# ------------------------------------------------------- activity degeneracy (1.4)
ctx_de = json.loads((WS / "reports/assets/gse283614_knockdown3/activity_context_de.json").read_text(encoding="utf-8"))
ctx_esc = json.loads((WS / "reports/assets/gse283614_knockdown3/activity_context_esc.json").read_text(encoding="utf-8"))
cov_de, cov_esc = ctx_de["coverage"], ctx_esc["coverage"]
check("DE activity axis: 31 evaluated, 19 measured, 12 written as 0, 10 zeros, 9 unique, median 0",
      cov_de["n_evaluated"] == 31 and cov_de["n_measured"] == 19
      and cov_de["n_unmeasured_written_as_zero"] == 12
      and cov_de["activity_distribution"]["n_zero"] == 10
      and cov_de["activity_distribution"]["n_unique"] == 9
      and cov_de["activity_distribution"]["median"] == 0.0)
check("ESC activity axis: 33 evaluated, 25 measured, 20 unique, median 0.571",
      cov_esc["n_evaluated"] == 33 and cov_esc["n_measured"] == 25
      and cov_esc["activity_distribution"]["n_unique"] == 20
      and close(cov_esc["activity_distribution"]["median"], 0.571086, 1e-5))
check("proxy rho: DE 0.333 p=0.163 (n=19) / ESC 0.575 p=0.0027 (n=25)",
      close(ctx_de["proxy_quality"]["overall"]["spearman_rho"], 0.3331, 0.001)
      and close(ctx_de["proxy_quality"]["overall"]["p_value"], 0.1635, 0.001)
      and close(ctx_esc["proxy_quality"]["overall"]["spearman_rho"], 0.5746, 0.001))
check("ESC proxy is driven by OCT4 (0.789) not NANOG (0.041)",
      close(ctx_esc["proxy_quality"]["within_target"]["OCT4"]["spearman_rho"], 0.7885, 0.001)
      and close(ctx_esc["proxy_quality"]["within_target"]["NANOG"]["spearman_rho"], 0.0410, 0.001))

# ------------------------------------------------- Tier B (only if it has run)
head = A / "rescore_headline_kd3.csv"
if head.exists():
    rh = keyed(pd.read_csv(head))
    full = rh[rh["variant"] == "full"]
    ref = keyed(pd.read_csv(A / "state_parallel.csv"))
    ref = ref[ref["design"] == "kd3"]
    for state in ("esc", "de"):
        for key in PRIMARY:
            got = full[(full["state"] == state) & (full["key"] == key)]
            want = ref[(ref["state"] == state) & (ref["key"] == key)]
            if got.empty or want.empty:
                continue
            check(f"Tier B 'full' variant reproduces published {state} {key}",
                  close(got.iloc[0]["rel_pct"], want.iloc[0]["relative_change_pct"], 0.01),
                  f"{got.iloc[0]['rel_pct']:.3f} vs {want.iloc[0]['relative_change_pct']:.3f}")

    def var(variant: str, state: str, key: str) -> pd.Series:
        return rh[(rh["variant"] == variant) & (rh["state"] == state) & (rh["key"] == key)].iloc[0]

    meta = json.loads((A / "rescore_kd3/rescore_meta.json").read_text(encoding="utf-8"))
    caps = meta.get("caps", meta)
    check("Tier B caps: count cap 38, control cap 415",
          int(caps.get("count", caps.get("count_cap", 0))) == 38
          and int(caps.get("control", caps.get("control_cap", 0))) == 415, str(caps))

    # count_matched: the DE penalty survives an 8x cell-count reduction
    check("T1.3 count_matched DE stays 6/6 worse",
          all(bool(var("count_matched", "de", k)["worse"]) for k in PRIMARY))
    check("T1.3 count_matched DE mse +19.3% / e_distance +5.3%",
          close(var("count_matched", "de", "mse@100")["rel_pct"], 19.27, 0.02)
          and close(var("count_matched", "de", "e_distance@100")["rel_pct"], 5.337, 0.02))
    r = var("count_matched", "de", "common_degs@cutoff")
    check("T1.3 count_matched DE common_degs -7.4% still BH significant (q=1.7e-4)",
          close(r["rel_pct"], -7.352, 0.02) and close(r["q_min"], 1.739e-4, 1e-5),
          f"{r['rel_pct']:.2f}% q={r['q_min']:.3g}")
    r = var("count_matched", "de", "pcc_delta@100")
    check("T1.3 count_matched collapses DE pcc_delta to -1.8% with unstable sign",
          close(r["rel_pct"], -1.759, 0.02) and r["rel_pct_min"] < 0 < r["rel_pct_max"],
          f"{r['rel_pct']:.2f}% range {r['rel_pct_min']:.1f}..{r['rel_pct_max']:.1f}")
    check("T1.3 count_matched ESC is only 2/6 worse (mse/pcc/wass/common_degs better)",
          sum(bool(var("count_matched", "esc", k)["worse"]) for k in PRIMARY) == 2)
    r = var("count_matched", "esc", "kl_divergence@100")
    check("T1.3 count_matched ESC kl +5.9% q=0.0022",
          close(r["rel_pct"], 5.934, 0.02) and close(r["q_min"], 2.247e-3, 1e-4))

    # control_matched: control pool size is irrelevant
    check("T1.4 control_matched DE reproduces full on mse/e_distance/wasserstein/kl",
          all(close(var("control_matched", "de", k)["rel_pct"], var("full", "de", k)["rel_pct"], 0.01)
              for k in ["mse@100", "e_distance@100", "wasserstein@100", "kl_divergence@100"]))
    check("T1.4 control_matched DE common_degs -14.9% (vs full -16.3%)",
          close(var("control_matched", "de", "common_degs@cutoff")["rel_pct"], -14.89, 0.02))

    # T4.2 per-stratum
    check("T4.2 DE high stratum: mse +9.9, pcc -11.5, common_degs -16.9 (5/6 worse)",
          close(var("stratum_high", "de", "mse@100")["rel_pct"], 9.919, 0.02)
          and close(var("stratum_high", "de", "pcc_delta@100")["rel_pct"], -11.54, 0.02)
          and close(var("stratum_high", "de", "common_degs@cutoff")["rel_pct"], -16.87, 0.02)
          and sum(bool(var("stratum_high", "de", k)["worse"]) for k in PRIMARY) == 5)
    check("T4.2 DE mid stratum: 2/6 worse, common_degs -8.6% q=0.007",
          sum(bool(var("stratum_mid", "de", k)["worse"]) for k in PRIMARY) == 2
          and close(var("stratum_mid", "de", "common_degs@cutoff")["rel_pct"], -8.58, 0.02)
          and close(var("stratum_mid", "de", "common_degs@cutoff")["q_min"], 7.017e-3, 1e-4))
    check("T4.2 DE low stratum: 5/6 worse, common_degs -6.5% q=0.038",
          sum(bool(var("stratum_low", "de", k)["worse"]) for k in PRIMARY) == 5
          and close(var("stratum_low", "de", "common_degs@cutoff")["rel_pct"], -6.457, 0.02)
          and close(var("stratum_low", "de", "common_degs@cutoff")["q_min"], 0.0378, 1e-3))
    check("T4.2 ESC high stratum: 1/6 worse, common_degs +33.8% q=0.008",
          sum(bool(var("stratum_high", "esc", k)["worse"]) for k in PRIMARY) == 1
          and close(var("stratum_high", "esc", "common_degs@cutoff")["rel_pct"], 33.83, 0.02)
          and close(var("stratum_high", "esc", "common_degs@cutoff")["q_min"], 8.392e-3, 1e-4))
    check("T4.2 ESC low stratum: 6/6 worse, mse +52.7%, pcc -36.3%, common_degs -30.2% q=0.024",
          sum(bool(var("stratum_low", "esc", k)["worse"]) for k in PRIMARY) == 6
          and close(var("stratum_low", "esc", "mse@100")["rel_pct"], 52.74, 0.02)
          and close(var("stratum_low", "esc", "pcc_delta@100")["rel_pct"], -36.28, 0.02)
          and close(var("stratum_low", "esc", "common_degs@cutoff")["rel_pct"], -30.2, 0.02)
          and close(var("stratum_low", "esc", "common_degs@cutoff")["q_min"], 0.02441, 1e-4))
    check("T4.2 ESC mid stratum: 2/6 worse, common_degs +3.6%",
          sum(bool(var("stratum_mid", "esc", k)["worse"]) for k in PRIMARY) == 2
          and close(var("stratum_mid", "esc", "common_degs@cutoff")["rel_pct"], 3.564, 0.02))

    # stratum_matched (T4.3), re-run with the fixed sampler
    sm_path = A / "rescore_headline_kd3_stratum.csv"
    if sm_path.exists():
        sm = keyed(pd.read_csv(sm_path))
        sm = sm[sm["variant"] == "stratum_matched"]

        def smv(state: str, key: str) -> pd.Series:
            return sm[(sm["state"] == state) & (sm["key"] == key)].iloc[0]

        check("T4.3 stratum_matched actually resamples (not a no-op)",
              any(not close(smv("de", k)["rel_pct"], var("full", "de", k)["rel_pct"], 0.005)
                  for k in PRIMARY))
        check("T4.3 DE still 6/6 worse after composition matching",
              sum(bool(smv("de", k)["worse"]) for k in PRIMARY) == 6)
        check("T4.3 DE: mse +19.3, pcc -9.2, common_degs -15.8 (q=9.6e-4)",
              close(smv("de", "mse@100")["rel_pct"], 19.28, 0.02)
              and close(smv("de", "pcc_delta@100")["rel_pct"], -9.25, 0.02)
              and close(smv("de", "common_degs@cutoff")["rel_pct"], -15.84, 0.02)
              and close(smv("de", "common_degs@cutoff")["q_min"], 9.62e-4, 1e-5))
        check("T4.3 DE composition accounts for pcc_delta's drop (only metric halved)",
              abs(smv("de", "pcc_delta@100")["rel_pct"]) < abs(var("full", "de", "pcc_delta@100")["rel_pct"]) / 2
              and abs(smv("de", "e_distance@100")["rel_pct"]) > abs(var("full", "de", "e_distance@100")["rel_pct"]) * 0.7)
        check("T4.3 ESC after matching: 2/6 worse, pcc +20.2, common_degs +4.9",
              sum(bool(smv("esc", k)["worse"]) for k in PRIMARY) == 2
              and close(smv("esc", "pcc_delta@100")["rel_pct"], 20.17, 0.02)
              and close(smv("esc", "common_degs@cutoff")["rel_pct"], 4.856, 0.02))
        lvl = pd.read_csv(A / "rescore_kd3_stratum/rescore_per_condition.csv")
        lv = lvl[(lvl["metric"] == "mse") & (lvl["deg_budget"].astype(str) == "100")]
        sizes = lv.groupby(["state", "resample_seed"])["n_imputed"].mean().groupby("state").mean()
        check("T4.3 composition matching drops 17% of DE cells and 40% of ESC cells",
              close(sizes["de"], 142.85, 0.2) and close(sizes["esc"], 63.29, 0.2),
              f"de={sizes['de']:.1f} esc={sizes['esc']:.1f} (full: de 171.8, esc 105.5)")
    else:
        check("Tier B stratum-matched rerun present (rescore_headline_kd3_stratum.csv)", False,
              "composition matching was a no-op in the first pass; rerun pending")
else:
    check("Tier B rescore present (rescore_headline_kd3.csv)", False, "run the container stage first")

print()
if FAIL:
    print(f"{len(FAIL)} claim(s) FAILED:")
    for name in FAIL:
        print(f"  - {name}")
    sys.exit(1)
print("all claims verified")
