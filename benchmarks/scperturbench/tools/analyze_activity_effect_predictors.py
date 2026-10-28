"""Does the activity conditioning help wherever there is more within-target guide activity variation?

For each evaluated condition, correlate the activity effect (relative change in mse / pcc_delta /
common_degs) against candidate predictors: within-target activity spread of the *training* guides,
the number of training guides, and the test-side activity of the evaluated guide.  If the effect is
concentrated where the activity axis carries signal, that is a mechanism, not noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/root/Diff-AE-Perturb") if Path("/root/Diff-AE-Perturb").exists() else Path(
    r"C:\Users\Admin\myprojects\Diff-AE-Perturb"
)
ROOT = Path(
    "/root/perturbench/.scperturbench/results/gse132080_activity_ablation/gse132080"
    if Path("/root/perturbench").exists()
    else r"C:\Users\Admin\myprojects\perturbench\.scperturbench\results"
    r"\gse132080_activity_ablation\gse132080"
)
SPLITS = Path(
    "/root/perturbench/.scperturbench/inputs/gse132080_squidiff_repro"
    if Path("/root/perturbench").exists()
    else r"C:\Users\Admin\myprojects\perturbench\.scperturbench\inputs\gse132080_squidiff_repro"
)
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    from perturbdiff.data import load_guide_activity_lookup

    table = REPO / "datasets/GSE132080/41587_2019_387_MOESM3_ESM/Table_S16_perturb-seq_screen_phenotypes.txt"
    if not table.exists():
        print(f"activity table not reachable at {table}; run inside the container")
        return 1
    activity = load_guide_activity_lookup(table, activity_col="relative_activity_day10")

    train = ad.read_h5ad(SPLITS / "squidiff_train.h5ad")
    tguides = train.obs["guide_identity"].astype(str)
    tlabels = train.obs["perturbation_label"].astype(str)

    on = pd.read_csv(ROOT / "activity_on_200ep/seed7/score/per_condition.csv")
    off = pd.read_csv(ROOT / "activity_off_200ep/seed7/score/per_condition.csv")

    # predictor table per condition
    rows = {}
    for condition in sorted(set(on["condition"])):
        guides = sorted({g for g, l in zip(tguides, tlabels) if l == condition and not l.startswith("control")})
        vals = [activity[g] for g in guides if g in activity]
        rows[condition] = {
            "n_train_guides": len(guides),
            "mean_train_activity": float(np.mean(vals)) if vals else np.nan,
            "range_train_activity": float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0,
            "std_train_activity": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    pred = pd.DataFrame(rows).T

    effects = {}
    for metric, budget in (("mse", "100"), ("pcc_delta", "100"), ("common_degs", "cutoff"), ("e_distance", "100")):
        a = on[(on["metric"] == metric) & (on["deg_budget"].astype(str) == budget)].set_index("condition")["value"]
        b = off[(off["metric"] == metric) & (off["deg_budget"].astype(str) == budget)].set_index("condition")["value"]
        rel = ((a - b) / b.replace(0, np.nan)).rename(f"rel_{metric}")
        effects[f"rel_{metric}"] = rel
    eff = pd.concat(effects.values(), axis=1)

    table_out = pred.join(eff)
    print(table_out.to_string(float_format=lambda v: f"{v:.4g}"))

    print("\n=== Spearman correlation (predictor vs relative effect) ===")
    print(f"{'predictor':<22} {'metric':<14} {'rho':>7} {'p':>9}")
    for predictor in ("n_train_guides", "mean_train_activity", "range_train_activity", "std_train_activity"):
        for metric in ("rel_mse", "rel_pcc_delta", "rel_common_degs", "rel_e_distance"):
            sub = table_out[[predictor, metric]].dropna()
            if sub.shape[0] < 5 or sub[predictor].nunique() < 2:
                continue
            rho, p = stats.spearmanr(sub[predictor], sub[metric])
            flag = "  <-- p<0.05" if p < 0.05 else ""
            print(f"{predictor:<22} {metric:<14} {rho:>7.3f} {p:>9.4f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
