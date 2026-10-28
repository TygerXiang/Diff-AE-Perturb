"""Why does the measured-activity channel reverse sign in DE (GSE283614)?

Read-only analysis over the artifacts the existing pipeline already wrote.  It
implements the pre-registered checks in
``.scperturbench/work/gse283614_de_vs_esc_predictions.md``:

Phase 0 (claim audit / H1)
  * state_parallel.csv        published on-vs-off table for both states, both designs
  * design_compare.csv        dose_cv vs kd3 per metric/state
  * jackknife_folds.csv       leave-one-fold-out recomputation (T1.1)
  * common_guides.csv         both states restricted to their shared guides (T1.2)
  * deg_overlap.csv           per-guide DEG-set overlap between states (T1.5)

Hypotheses
  * per_target_bands.csv      target x guide_type x activity-band splits (H2, H6, T7.3)
  * selective_pressure*.csv   dose/activity/cell-count vs per-guide harm (H5, T5.1)
  * probe_compare.csv         activity-swap probe, both states/designs (H7, T7.1)
  * probe_vs_harm.csv         per-guide probe shift vs per-guide metric damage (T7.2)
  * curve_fits_de.csv         linear/quadratic/Hill fits of the paper's state scores (H3, T3.1)
  * summary.json              the headline numbers the report and the claim checker quote

Nothing is trained and nothing outside ``--output-dir`` is written.  Every
statistic that the published reports already report is echoed from their own
CSVs (tagged ``source=published``) so this analysis can never silently disagree
with them; recomputed numbers carry the subset they were computed on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
WS = HERE.parent
DIFFAE = Path(r"C:\Users\Admin\myprojects\Diff-AE-Perturb")
BIO = DIFFAE / "runs/biological_validation/gse283614_observed"

PRIMARY = [("mse", "100"), ("pcc_delta", "100"), ("e_distance", "100"),
           ("wasserstein", "100"), ("kl_divergence", "100"), ("common_degs", "cutoff")]
EXTRA = [("mse", "all"), ("pcc_delta", "all"), ("e_distance", "all"), ("kl_divergence", "all")]
ALL_CELLS = PRIMARY + EXTRA
PRIMARY_LABEL = {m: f"{m}@{b}" for m, b in PRIMARY}

DESIGNS = {
    "dose_cv": {
        "results": WS / ".scperturbench/results/gse283614_activity_ablation/gse283614",
        "inputs": WS / ".scperturbench/inputs/gse283614_activity_ablation",
        "deg_pattern": "deg_table_{state}_fold{fold}.json",
        "assets": WS / "reports/assets/gse283614_activity_ablation",
        "h5ad": DIFFAE / "reproducibility/prepared/gse283614/dose_cv",
    },
    "kd3": {
        "results": WS / ".scperturbench/results/gse283614_activity_ablation_kd3/gse283614",
        "inputs": WS / ".scperturbench/inputs/gse283614_knockdown3_inputs",
        "deg_pattern": "deg_table_kd3_{state}_fold{fold}.json",
        "assets": WS / "reports/assets/gse283614_knockdown3",
        "h5ad": WS / ".scperturbench/inputs/gse283614_knockdown3",
    },
}
STATES = ["esc", "de"]


# --------------------------------------------------------------------------- #
# small statistics helpers
# --------------------------------------------------------------------------- #
def bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q values, NaNs preserved (no NaN -> no rank)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    idx = np.argsort(p[finite])
    ranked = p[finite][idx]
    n = ranked.size
    if n == 0:
        return q
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n)
    out[idx] = np.clip(adjusted, 0.0, 1.0)
    q[finite] = out
    return q


def hedges_g(deltas: np.ndarray) -> float:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return float("nan")
    g = d.mean() / sd
    return float(g * (1 - 3 / (4 * n - 5)))


def paired_stats(deltas: np.ndarray) -> dict:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    out = {"n": int(n), "p_wilcoxon": float("nan"), "p_t": float("nan"),
           "hedges_g": hedges_g(d)}
    if n == 0:
        return out
    if np.any(d != 0):
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            pass
    if n >= 2 and d.std(ddof=1) > 0:
        out["p_t"] = float(stats.ttest_1samp(d, 0.0).pvalue)
    return out


def relative_pct(mean_on: float, mean_off: float) -> float:
    if mean_off == 0 or not np.isfinite(mean_off):
        return float("nan")
    return 100.0 * (mean_on - mean_off) / mean_off


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def summarize(frame: pd.DataFrame, subset: str, source: str,
              group_cols: list[str] | None = None) -> pd.DataFrame:
    """Per metric rows: paired on-vs-off stats over one subset of conditions."""
    rows = []
    for (metric, budget), block in frame.groupby(["metric", "deg_budget"], sort=False):
        key = {"metric": metric, "deg_budget": str(budget), "subset": subset, "source": source}
        if group_cols:
            for col in group_cols:
                key[col] = block[col].iloc[0]
        deltas = block["paired_delta_on_minus_off"].to_numpy(dtype=float)
        stat = paired_stats(deltas)
        mean_on = float(block["value_activity_on"].mean())
        mean_off = float(block["value_activity_off"].mean())
        mean_abs_off = float(block["value_activity_off"].abs().mean())
        higher = bool(block["higher_is_better"].iloc[0])
        rel = relative_pct(mean_on, mean_off)
        robust = relative_pct(mean_on, mean_abs_off)
        key.update({
            "n_conditions": stat["n"], "mean_on": mean_on, "mean_off": mean_off,
            "mean_abs_off": mean_abs_off,
            "relative_change_pct": rel,
            "relative_change_pct_robust": robust,
            "degenerate_denominator": bool(mean_abs_off > 0
                                           and abs(mean_off) < 0.2 * mean_abs_off),
            "p_wilcoxon": stat["p_wilcoxon"],
            "p_t": stat["p_t"], "hedges_g": stat["hedges_g"],
            "higher_is_better": higher,
            "better": bool((rel > 0) == higher) if np.isfinite(rel) else False,
        })
        rows.append(key)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {f"{m}@{b}": i for i, (m, b) in enumerate(ALL_CELLS)}
    out["_order"] = out.apply(lambda r: order.get(f"{r['metric']}@{r['deg_budget']}", 99), axis=1)
    out = out.sort_values(["_order"] + [c for c in (group_cols or [])]).drop(columns="_order")
    out["q_bh"] = bh(out["p_wilcoxon"].to_numpy(dtype=float))
    return out.reset_index(drop=True)


def load_design(design: str, state: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = DESIGNS[design]
    base = cfg["results"] / state / "ablation_analysis"
    per_cond = pd.read_csv(base / "ablation_per_condition.csv")
    per_cond["metric_deg"] = per_cond["metric"] + "@" + per_cond["deg_budget"].astype(str)
    per_cond = per_cond[per_cond["metric_deg"].isin([f"{m}@{b}" for m, b in ALL_CELLS])].copy()
    per_cond["activity_measured"] = as_bool(per_cond["activity_measured"])
    per_cond["higher_is_better"] = as_bool(per_cond["higher_is_better"])
    published = pd.read_csv(base / "ablation_tests.csv")
    published["metric_deg"] = published["metric"] + "@" + published["deg_budget"].astype(str)
    return per_cond, published


# --------------------------------------------------------------------------- #
# Phase 0
# --------------------------------------------------------------------------- #
def state_parallel() -> pd.DataFrame:
    rows = []
    for design in DESIGNS:
        for state in STATES:
            _, published = load_design(design, state)
            for _, r in published.iterrows():
                if r["metric_deg"] not in {f"{m}@{b}" for m, b in ALL_CELLS}:
                    continue
                rows.append({
                    "design": design, "state": state,
                    "metric": r["metric"], "deg_budget": str(r["deg_budget"]),
                    "n_conditions": int(r["n_conditions"]),
                    "mean_on": float(r["mean_activity_on"]),
                    "mean_off": float(r["mean_activity_off"]),
                    "relative_change_pct": float(r["relative_change_pct"]),
                    "p_wilcoxon": float(r["wilcoxon_p"]),
                    "q_bh": float(r["wilcoxon_q_bh"]),
                    "p_t": float(r["paired_t_p"]),
                    "hedges_g": float(r["hedges_g_paired"]),
                    "higher_is_better": bool(r["higher_is_better"]),
                    "better": bool((r["relative_change_pct"] > 0) == bool(r["higher_is_better"])),
                    "source": "published",
                })
    out = pd.DataFrame(rows)
    order = {f"{m}@{b}": i for i, (m, b) in enumerate(ALL_CELLS)}
    out["_order"] = out["metric"].str.cat(out["deg_budget"], sep="@").map(order)
    return out.sort_values(["design", "state", "_order"]).drop(columns="_order").reset_index(drop=True)


def design_compare(parallel: pd.DataFrame) -> pd.DataFrame:
    wide = parallel.pivot_table(
        index=["metric", "deg_budget", "state"], columns="design",
        values="relative_change_pct", aggfunc="first").reset_index()
    wide = wide.rename(columns={"dose_cv": "rel_pct_dose_cv", "kd3": "rel_pct_kd3"})
    wide["sign_agree"] = np.sign(wide["rel_pct_dose_cv"]) == np.sign(wide["rel_pct_kd3"])
    return wide


def jackknife(per_cond_all: dict) -> pd.DataFrame:
    rows = []
    for design in DESIGNS:
        for state in STATES:
            frame = per_cond_all[(design, state)]
            for (metric, budget), block in frame.groupby(["metric", "deg_budget"], sort=False):
                folds = sorted(block["fold"].unique())
                variants = [("all", block)] + [
                    (f"drop_fold_{f}", block[block["fold"] != f]) for f in folds
                ]
                for label, sub in variants:
                    stat = paired_stats(sub["paired_delta_on_minus_off"].to_numpy(dtype=float))
                    mean_on = float(sub["value_activity_on"].mean())
                    mean_off = float(sub["value_activity_off"].mean())
                    mean_abs_off = float(sub["value_activity_off"].abs().mean())
                    rel = relative_pct(mean_on, mean_off)
                    rows.append({
                        "design": design, "state": state, "metric": metric,
                        "deg_budget": str(budget), "variant": label,
                        "n_conditions": stat["n"],
                        "mean_on": mean_on,
                        "mean_off": mean_off,
                        "relative_change_pct": rel,
                        "relative_change_pct_robust": relative_pct(mean_on, mean_abs_off),
                        "p_wilcoxon": stat["p_wilcoxon"],
                        "better": bool((rel > 0) == bool(sub["higher_is_better"].iloc[0]))
                        if np.isfinite(rel) else False,
                    })
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    for (design, state, variant), block in out.groupby(["design", "state", "variant"]):
        out.loc[block.index, "q_bh"] = bh(block["p_wilcoxon"].to_numpy(dtype=float))
    return out.reset_index(drop=True)


def common_guides(per_cond_all: dict) -> pd.DataFrame:
    rows = []
    for design in DESIGNS:
        esc = per_cond_all[(design, "esc")]
        de = per_cond_all[(design, "de")]
        shared = sorted(set(esc["condition"]) & set(de["condition"]))
        for state, frame in (("esc", esc), ("de", de)):
            sub = frame[frame["condition"].isin(shared)]
            block = summarize(sub, subset=f"shared_guides_n{len(shared)}",
                              source="recomputed")
            if block.empty:
                continue
            block.insert(0, "state", state)
            block.insert(0, "design", design)
            block["n_shared_guides"] = len(shared)
            rows.append(block)
    return pd.concat(rows, ignore_index=True)


def deg_overlap(design: str) -> pd.DataFrame:
    cfg = DESIGNS[design]
    per_state: dict[str, dict[str, set[str]]] = {}
    for state in STATES:
        merged: dict[str, set[str]] = {}
        for path in sorted(cfg["inputs"].glob(f"*{state}*fold*.json")):
            if not path.name.startswith("deg_table"):
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for guide, payload in data.get("genes", {}).items():
                if isinstance(payload, dict):
                    gene_list = list(payload.get("order", []))
                else:
                    gene_list = list(payload)
                merged.setdefault(guide, set(gene_list[:100]))
        per_state[state] = merged
    rows = []
    for guide in sorted(set(per_state["esc"]) & set(per_state["de"])):
        a, b = per_state["esc"][guide], per_state["de"][guide]
        union = a | b
        rows.append({
            "design": design, "guide": guide, "n_deg_esc": len(a), "n_deg_de": len(b),
            "n_overlap": len(a & b),
            "jaccard": len(a & b) / len(union) if union else float("nan"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["overlap_fraction_of_100"] = out["n_overlap"] / 100.0
    return out


# --------------------------------------------------------------------------- #
# H2 / H6 / H7.3 - subsets
# --------------------------------------------------------------------------- #
def activity_band(frame: pd.DataFrame) -> pd.Series:
    measured = frame["activity_measured"]
    value = pd.to_numeric(frame["activity_input"], errors="coerce").fillna(0.0)
    band = pd.Series("measured_nonzero", index=frame.index, dtype=object)
    band[value <= 0] = "measured_zero"
    band[~measured] = "unmeasured"
    return band


def dose_band(frame: pd.DataFrame) -> pd.Series:
    """Tertiles of the paper's measured knockdown fraction, per design.

    Used as the pre-registered T3.4 surrogate: guides beyond the response vertex are
    expected in the top tertile, so if a monotone strength channel is the problem the
    top band should carry the negative effect.
    """
    dose = pd.to_numeric(frame["knockdown_fraction"], errors="coerce")
    band = pd.Series("unknown", index=frame.index, dtype=object)
    finite = dose.dropna()
    if finite.nunique() < 3:
        return band
    edges = finite.quantile([1 / 3, 2 / 3]).to_numpy()
    band[dose.notna()] = np.array(["low", "mid", "high"])[
        np.digitize(dose[dose.notna()].to_numpy(), edges)]
    return band


def per_target_bands(per_cond_all: dict) -> pd.DataFrame:
    rows = []
    dose = (pd.read_csv(BIO / "gse283614_tf_dosage.tsv", sep="\t")
            .rename(columns={"guide_identity": "condition"})
            [["condition", "knockdown_fraction", "tf_remaining"]]
            .drop_duplicates("condition"))
    for design in DESIGNS:
        for state in STATES:
            frame = per_cond_all[(design, state)].merge(dose, on="condition", how="left")
            frame["band"] = activity_band(frame)
            frame["dose_band"] = dose_band(frame)
            splits = {
                "target": "target_gene",
                "guide_type": "guide_type",
                "activity_band": "band",
                "dose_band": "dose_band",
            }
            for group_type, col in splits.items():
                for _, sub in frame.groupby(col, sort=True):
                    block = summarize(sub, subset=f"{group_type}={sub[col].iloc[0]}",
                                      source="recomputed")
                    if block.empty:
                        continue
                    block.insert(0, "state", state)
                    block.insert(0, "design", design)
                    block["group_type"] = group_type
                    block["group"] = str(sub[col].iloc[0])
                    rows.append(block)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# H5 / T5.1 - selective pressure
# --------------------------------------------------------------------------- #
def ground_truth_covariates() -> pd.DataFrame:
    """Per guide per state per fold: cell count plus count/MT covariates."""
    import anndata as ad

    rows = []
    for design, cfg in DESIGNS.items():
        for state in STATES:
            root = cfg["h5ad"] / state
            for fold_dir in sorted(root.glob("fold_*")):
                test = fold_dir / f"gse283614_{state}_test.h5ad"
                if not test.exists():
                    continue
                fold = int(fold_dir.name.split("_")[1])
                obs = ad.read_h5ad(test, backed="r").obs
                for guide, block in obs.groupby("guide_identity", sort=True, observed=True):
                    rows.append({
                        "design": design, "state": state, "fold": fold,
                        "guide": str(guide),
                        "n_gt_cells": int(len(block)),
                        "mean_total_counts": float(block["total_counts"].mean()),
                        "mean_pct_mt": float(block["pct_counts_mt"].mean()),
                        "mean_n_genes": float(block["n_genes_by_counts"].mean()),
                        "mean_guide_umis": float(block["guide_total_umis"].mean()),
                    })
    return pd.DataFrame(rows)


def selective_pressure(per_cond_all: dict, gt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dose = pd.read_csv(BIO / "gse283614_tf_dosage.tsv", sep="\t")
    rows = []
    for design in DESIGNS:
        for state in STATES:
            frame = per_cond_all[(design, state)]
            pivot = frame.pivot_table(
                index="condition", columns="metric_deg",
                values="paired_delta_on_minus_off", aggfunc="first")
            meta = frame.groupby("condition").agg(
                target_gene=("target_gene", "first"), guide_type=("guide_type", "first"),
                activity_input=("activity_input", "first"),
                activity_measured=("activity_measured", "first"),
                n_test_cells=("n_test_cells", "first"), fold=("fold", "first")).reset_index()
            merged = meta.merge(pivot, on="condition", how="left")
            merged["design"] = design
            merged["state"] = state
            rows.append(merged.rename(columns={"condition": "guide"}))
    per_guide = pd.concat(rows, ignore_index=True)
    per_guide = per_guide.merge(dose[["guide_identity", "knockdown_fraction", "tf_remaining"]],
                                left_on="guide", right_on="guide_identity", how="left")
    per_guide = per_guide.drop(columns=["guide_identity"])
    per_guide = per_guide.merge(gt, on=["design", "state", "guide"], how="left")
    per_guide["activity_measured"] = as_bool(per_guide["activity_measured"])

    corr_rows = []
    predictors = ["knockdown_fraction", "activity_input", "n_gt_cells",
                  "mean_total_counts", "mean_pct_mt", "mean_guide_umis"]
    for (design, state), block in per_guide.groupby(["design", "state"]):
        for pred in predictors:
            for target in ["n_gt_cells", "mean_total_counts", "mean_pct_mt"] + \
                    [f"{m}@{b}" for m, b in PRIMARY]:
                if pred == target or target not in block.columns:
                    continue
                sub = block[[pred, target]].dropna()
                if len(sub) < 5 or sub[pred].nunique() < 3 or sub[target].nunique() < 3:
                    continue
                rho, p = stats.spearmanr(sub[pred], sub[target])
                corr_rows.append({"design": design, "state": state, "x": pred,
                                  "y": target, "spearman_rho": float(rho),
                                  "p_value": float(p), "n": int(len(sub))})
    corr = pd.DataFrame(corr_rows)
    if not corr.empty:
        corr["q_bh"] = np.nan
        for (design, state), block in corr.groupby(["design", "state"]):
            corr.loc[block.index, "q_bh"] = bh(block["p_value"].to_numpy(dtype=float))
    return per_guide, corr


# --------------------------------------------------------------------------- #
# H7 - probe
# --------------------------------------------------------------------------- #
def probe_compare(per_cond_all: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for design, cfg in DESIGNS.items():
        path = cfg["assets"] / "probe_all.csv"
        if path.exists():
            d = pd.read_csv(path)
            d["design"] = design
            frames.append(d)
    # T7.1: the DE probe was originally run with 4 samples while ESC used 8, which
    # inflates the DE noise floor by ~sqrt(2).  A re-run with 8 samples lives in its
    # own results subtree; load it as a separate probe design so both are reported.
    n8_root = WS / ".scperturbench/results/gse283614_de_probe8"
    n8_files = sorted(n8_root.glob("*/fold_*/activity_on/activity_perturbation_probe_seed*.csv"))
    if n8_files:
        n8 = pd.concat([pd.read_csv(p) for p in n8_files], ignore_index=True)
        n8["design"] = "kd3_de_n8"
        frames.append(n8)
    probe = pd.concat(frames, ignore_index=True)
    rows = []
    for (design, state), block in probe.groupby(["design", "state"]):
        swap = block[np.isfinite(block["activity_delta"])]
        changed = swap[swap["activity_delta"].abs() > 1e-12]
        zero = swap[swap["activity_delta"].abs() <= 1e-12]
        hit_zero = changed[np.isclose(changed["activity_swapped"], 0.0)]
        nonzero = changed[~np.isclose(changed["activity_swapped"], 0.0)]
        floor = float(changed["l2_noise_floor"].median()) if len(changed) else float("nan")
        med_shift = float(changed["l2_shift_swapped"].median()) if len(changed) else float("nan")
        rows.append({
            "design": design, "state": state,
            "n_swaps": int(len(swap)), "n_changed": int(len(changed)),
            "n_identical_value_swaps": int(len(zero)),
            "max_shift_of_identical_swaps": float(zero["l2_shift_swapped"].max()) if len(zero) else float("nan"),
            "median_shift_changed": med_shift,
            "median_noise_floor": floor,
            "ratio_shift_over_noise": med_shift / floor if floor else float("nan"),
            "n_hit_zero": int(len(hit_zero)),
            "hit_zero_median_shift": float(hit_zero["l2_shift_swapped"].median()) if len(hit_zero) else float("nan"),
            "hit_zero_ratio": (float(hit_zero["l2_shift_swapped"].median()) / floor)
            if len(hit_zero) and floor else float("nan"),
            "n_nonzero_swap": int(len(nonzero)),
            "nonzero_median_shift": float(nonzero["l2_shift_swapped"].median()) if len(nonzero) else float("nan"),
            "nonzero_ratio": (float(nonzero["l2_shift_swapped"].median()) / floor)
            if len(nonzero) and floor else float("nan"),
            "n_exceed_noise_floor": int((changed["l2_shift_swapped"] > changed["l2_noise_floor"]).sum()),
            "median_between_guide_l2": float(changed["l2_between_guide"].median()) if len(changed) else float("nan"),
            "n_samples": int(block["n_samples"].max()),
        })
    summary = pd.DataFrame(rows)

    # per-guide probe shift (median over folds) vs per-guide damage in kd3.
    # Stratified by measurement status: only a *changed* swap can move the
    # prediction at all, and measured guides are the ones that improve, so the
    # pooled correlation is confounded unless the two groups are also reported.
    rows = []
    probe_designs = [d for d in probe["design"].unique() if d in DESIGNS or d == "kd3_de_n8"]
    for probe_design in probe_designs:
        results_design = "kd3" if probe_design == "kd3_de_n8" else probe_design
        for state in STATES:
            sub = probe[(probe["design"] == probe_design) & (probe["state"] == state)]
            if sub.empty:
                continue
            per_guide = sub.groupby("guide", observed=True).agg(
                shift=("l2_shift_swapped", "median"),
                noise=("l2_noise_floor", "median"),
                between=("l2_between_guide", "median"),
                activity_true=("activity_true", "median")).reset_index()
            per_guide["ratio"] = per_guide["shift"] / per_guide["noise"]
            per_guide["ratio_between"] = per_guide["shift"] / per_guide["between"]
            frame = per_cond_all[(results_design, state)]
            for (metric, budget), block in frame.groupby(["metric", "deg_budget"], sort=False):
                merged = per_guide.merge(
                    block[["condition", "paired_delta_on_minus_off", "higher_is_better",
                           "activity_measured", "activity_input"]],
                    left_on="guide", right_on="condition", how="inner")
                merged["activity_measured"] = as_bool(merged["activity_measured"])
                merged = merged.dropna(subset=["ratio", "paired_delta_on_minus_off"])
                higher = bool(merged["higher_is_better"].iloc[0])
                merged["damage"] = merged["paired_delta_on_minus_off"] * (-1 if higher else 1)
                scopes = {
                    "all_changed_swaps": merged,
                    "measured": merged[merged["activity_measured"]],
                    "unmeasured": merged[~merged["activity_measured"]],
                }
                for scope, block2 in scopes.items():
                    for ratio_col in ("ratio", "ratio_between"):
                        pair = block2[[ratio_col, "damage"]].dropna()
                        pair = pair[np.isfinite(pair[ratio_col])]
                        if len(pair) < 6 or pair[ratio_col].nunique() < 3:
                            continue
                        rho, p = stats.spearmanr(pair[ratio_col], pair["damage"])
                        rows.append({"design": results_design, "probe_design": probe_design,
                                     "probe_n_samples": int(sub["n_samples"].max()),
                                     "state": state, "metric": metric,
                                     "deg_budget": str(budget), "scope": scope,
                                     "ratio_column": ratio_col, "n": int(len(pair)),
                                     "spearman_rho_ratio_vs_damage": float(rho), "p_value": float(p)})
    harm = pd.DataFrame(rows)
    if not harm.empty:
        harm["q_bh"] = np.nan
        for (design, state), block in harm.groupby(["design", "state"]):
            harm.loc[block.index, "q_bh"] = bh(block["p_value"].to_numpy(dtype=float))
    return summary, harm


# --------------------------------------------------------------------------- #
# H3 / T3.1 - response-curve shape in the paper's own scores
# --------------------------------------------------------------------------- #
def _fit_and_select(x: np.ndarray, y: np.ndarray, models: tuple[str, ...]) -> dict:
    """Fit each model on the dose axis; select by the repo's Delta-AIC>2 rule."""
    n = len(x)
    fits: dict[str, dict] = {}
    for model in models:
        if model == "linear":
            k = 2
            coeffs = np.polyfit(x, y, 1)
            pred = np.polyval(coeffs, x)
            extra = {"slope": float(coeffs[0]), "intercept": float(coeffs[1])}
        elif model == "quadratic":
            k = 3
            coeffs = np.polyfit(x, y, 2)
            pred = np.polyval(coeffs, x)
            a, b, c = coeffs
            vertex = -b / (2 * a) if a != 0 else float("nan")
            extra = {"quadratic": float(a), "linear": float(b), "intercept": float(c),
                     "vertex": float(vertex),
                     "inverted_u": bool(a < 0 and np.isfinite(vertex)
                                        and x.min() <= vertex <= x.max())}
        else:  # hill, the same monotone-decreasing family the repo fits on tf_remaining
            k = 2

            def hill(xx, ed50, h):
                xx = np.clip(xx, 1e-6, None)
                return 1.0 / (1.0 + (xx / max(ed50, 1e-6)) ** h)

            try:
                popt, _ = curve_fit(hill, x, y, p0=(0.5, 2.0),
                                    bounds=([1e-4, 0.05], [10.0, 50.0]), maxfev=20000)
                pred = hill(x, *popt)
                extra = {"ed50": float(popt[0]), "hill_coefficient": float(popt[1])}
            except Exception:
                continue
        rss = float(np.sum((y - pred) ** 2))
        aic = n * np.log(max(rss / n, 1e-12)) + 2 * k
        aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
        fits[model] = {"model": model, "k": k, "rss": rss, "aic": aic, "aicc": aicc, **extra}
    if "linear" not in fits:
        return {}
    aic_lin = fits["linear"]["aic"]
    # Same precedence as the paper / repo: the Hill model is only unseated by the
    # quadratic when the quadratic is better by more than the AIC margin.
    d_hill = aic_lin - fits["hill"]["aic"] if "hill" in fits else float("-inf")
    d_quad = aic_lin - fits["quadratic"]["aic"] if "quadratic" in fits else float("-inf")
    selected = "linear"
    if d_hill > 2 and d_hill >= d_quad:
        selected = "hill"
    elif d_quad > 2:
        selected = "quadratic"
    return {
        "selected": selected,
        "fits": fits,
        "delta_aic_linear_minus_quadratic": d_quad,
        "delta_aic_linear_minus_hill": d_hill,
    }


def curve_fits() -> pd.DataFrame:
    """Fit the paper's per-guide state scores against remaining TF level.

    The axis is ``tf_remaining`` (= 1 - knockdown_fraction), i.e. the same axis the
    paper and ``perturbchdiff.gse283614_biology`` use, so the Hill family is
    monotone-decreasing and the quadratic vertex is comparable with the repo's own
    ``author_curve_fits.csv`` (echoed as ``repo_*`` columns).
    """
    scores = pd.read_csv(BIO / "observed_guide_state_scores.csv")
    summary = json.loads((BIO / "observed_biology_summary.json").read_text(encoding="utf-8"))
    paper_reported = summary.get("paper_reported_models", {})
    repo_fits = pd.read_csv(BIO / "author_curve_fits.csv")
    repo_selected, repo_vertex = {}, {}
    for _, r in repo_fits.iterrows():
        if bool(r.get("selected_aic", False)):
            repo_selected[str(r["response"])] = str(r["model"])
            if str(r["model"]) == "quadratic":
                repo_vertex[str(r["response"])] = float(r["vertex"])
    responses = {
        "ESC": [("esc_score", "NANOG", "ESC_cluster1_score_vs_NANOG"),
                ("esc_score", "OCT4", "ESC_cluster2_score_vs_OCT4")],
        "DE": [("de_differentiation_score", "NANOG", "DE_differentiation_score_vs_NANOG"),
               ("de_differentiation_score", "OCT4", "DE_differentiation_score_vs_OCT4"),
               ("de_off_target_score", "NANOG", "DE_off-target_score_vs_NANOG"),
               ("de_off_target_score", "OCT4", "DE_off-target_score_vs_OCT4")],
    }
    rows = []
    for state, specs in responses.items():
        for column, target, label in specs:
            sub = scores[(scores["state"] == state) & (scores["dose_target"] == target)
                         & (~as_bool(scores["is_control"]))].dropna(
                subset=[column, "knockdown_fraction"])
            if len(sub) < 6:
                continue
            fit = _fit_and_select(sub["tf_remaining"].to_numpy(dtype=float),
                                  sub[column].to_numpy(dtype=float),
                                  ("linear", "quadratic", "hill"))
            if not fit:
                continue
            best = fit["fits"][fit["selected"]]
            vertex = best.get("vertex", np.nan)
            rows.append({
                "response": label, "state": state, "target": target,
                "score_column": column, "n_guides": int(len(sub)),
                "axis": "tf_remaining",
                "dose_min_tf_remaining": float(sub["tf_remaining"].min()),
                "dose_max_tf_remaining": float(sub["tf_remaining"].max()),
                "selected_model": fit["selected"],
                "aic_linear": fit["fits"]["linear"]["aic"],
                "delta_aic_linear_minus_quadratic": fit["delta_aic_linear_minus_quadratic"],
                "delta_aic_linear_minus_hill": fit["delta_aic_linear_minus_hill"],
                "quadratic_coef": best.get("quadratic", np.nan),
                "vertex_tf_remaining": vertex,
                "vertex_knockdown_fraction": (1.0 - vertex) if np.isfinite(vertex) else np.nan,
                "inverted_u": bool(best.get("inverted_u", False)),
                "ed50_tf_remaining": best.get("ed50", np.nan),
                "hill_coefficient": best.get("hill_coefficient", np.nan),
                "paper_reported_model": paper_reported.get(label, ""),
                "repo_selected_model": repo_selected.get(label, ""),
                "repo_vertex_tf_remaining": repo_vertex.get(label, np.nan),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["agrees_with_paper"] = out["selected_model"] == out["paper_reported_model"]
        out["agrees_with_repo_fit"] = out["selected_model"] == out["repo_selected_model"]
        out["repo_bootstrap_frequency"] = out["response"].map(
            lambda r: next((c.get("bootstrap", {}).get("selection_frequency", {})
                            for c in summary.get("author_response_curves", [])
                            if c.get("response") == r), {})).astype(str)
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=WS / "reports/assets/gse283614_de_vs_esc")
    args = parser.parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    global per_cond_all
    per_cond_all = {(d, s): load_design(d, s)[0] for d in DESIGNS for s in STATES}

    written: list[str] = []

    def dump(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        path = out_dir / name
        frame.to_csv(path, index=False)
        written.append(name)
        print(f"  wrote {name} ({len(frame)} rows)")
        return frame

    print("[phase 0] claim audit")
    parallel = dump(state_parallel(), "state_parallel.csv")
    dump(design_compare(parallel), "design_compare.csv")
    jk = dump(jackknife(per_cond_all), "jackknife_folds.csv")
    dump(common_guides(per_cond_all), "common_guides.csv")
    dump(pd.concat([deg_overlap(d) for d in DESIGNS], ignore_index=True), "deg_overlap.csv")

    print("[H2/H6] subsets")
    dump(per_target_bands(per_cond_all), "per_target_bands.csv")

    print("[H5] selective pressure")
    gt = ground_truth_covariates()
    per_guide, corr = selective_pressure(per_cond_all, gt)
    dump(per_guide, "selective_pressure.csv")
    dump(corr, "selective_pressure_corr.csv")

    print("[H7] probe")
    probe, harm = probe_compare(per_cond_all)
    dump(probe, "probe_compare.csv")
    dump(harm, "probe_vs_harm.csv")

    print("[H3] response-curve shape")
    dump(curve_fits(), "curve_fits_de.csv")

    # headline JSON for the report and the claim checker
    def cell(frame: pd.DataFrame, **kw) -> dict:
        sub = frame.copy()
        for key, value in kw.items():
            sub = sub[sub[key].astype(str) == str(value)]
        return sub.iloc[0].to_dict() if len(sub) else {}

    de_kd3 = cell(parallel, design="kd3", state="de", metric="common_degs", deg_budget="cutoff")
    esc_kd3 = cell(parallel, design="kd3", state="esc", metric="common_degs", deg_budget="cutoff")
    de_dc = cell(parallel, design="dose_cv", state="de", metric="e_distance", deg_budget="100")
    esc_dc = cell(parallel, design="dose_cv", state="esc", metric="pcc_delta", deg_budget="100")
    probe_esc = cell(probe, design="kd3", state="esc")
    probe_de = cell(probe, design="kd3", state="de")
    summary = {
        "claim": "the measured-activity channel reverses sign in DE vs ESC",
        "primary_effect": {
            "kd3_de_common_degs_pct": de_kd3.get("relative_change_pct"),
            "kd3_de_common_degs_q": de_kd3.get("q_bh"),
            "kd3_esc_common_degs_pct": esc_kd3.get("relative_change_pct"),
            "dose_cv_de_e_distance_pct": de_dc.get("relative_change_pct"),
            "dose_cv_de_e_distance_q": de_dc.get("q_bh"),
            "dose_cv_esc_pcc_delta_pct": esc_dc.get("relative_change_pct"),
        },
        "baseline_absolute": {
            "kd3_de_off_common_degs": de_kd3.get("mean_off"),
            "kd3_esc_off_common_degs": esc_kd3.get("mean_off"),
        },
        "probe": {"esc": probe_esc, "de": probe_de},
        "files": written,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  wrote summary.json")
    print(f"\noutput: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
