"""H4/H5: how much of the DE ground truth is "a mixture", and does dose shift it?

The paper's DE Perturb-seq is a *mid-transition* snapshot (36 h after DE induction)
with a continuous trajectory plus a low-ESC/low-DE cluster.  If each guide's ground
truth is really a mixture of cell states, then any model that emits one profile per
guide is mis-specified, and a distance-based metric will punish it in proportion to
how much the mixture moves.  This tool measures, on the ground-truth test cells only
(no model involved):

* ``heterogeneity_decomposition.csv`` - share of the total (per-gene centered)
  variance that is between guides, between strata inside a guide, and residual.
* ``stratum_composition.csv`` - per guide: cell count, mean state score, and the
  low/mid/high stratum fractions, joined to the paper's knockdown fraction.
* ``stratum_composition_corr.csv`` - Spearman of the composition fractions with
  dose/activity, and with the per-guide metric damage of the activity arm.

Strata are built from the paper's own gene sets (media-3.xlsx, Table S3): the DE
score for DE cells and the pluripotency score for ESC cells, split into tertiles.
Nothing is trained; only the prepared test h5ads and the RNA matrices are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
WS = HERE.parent
DIFFAE = Path(r"C:\Users\Admin\myprojects\Diff-AE-Perturb")
GENE_SET_WORKBOOK = DIFFAE / "datasets/GSE283614/media-3.xlsx"

DESIGNS = {
    "dose_cv": DIFFAE / "reproducibility/prepared/gse283614/dose_cv",
    "kd3": WS / ".scperturbench/inputs/gse283614_knockdown3",
}
STATES = ["esc", "de"]
# Which paper gene set defines each state's own axis.
STATE_SCORE_SET = {"esc": "Pluripotency genes", "de": "DE genes"}
# Extra axes scored in both states: the paper's DE trajectory and the off-target
# NE programme (its Cluster 2 is low-ESC/low-DE and NE-high).
EXTRA_SETS = ["DE genes", "NE genes"]
PRIMARY = [("mse", "100"), ("pcc_delta", "100"), ("e_distance", "100"),
           ("wasserstein", "100"), ("kl_divergence", "100"), ("common_degs", "cutoff")]


def read_gene_sets(path: Path) -> dict[str, list[str]]:
    raw = pd.read_excel(path, header=None)
    header_row = None
    for i in range(min(10, len(raw))):
        values = [str(v).strip() for v in raw.iloc[i].tolist()]
        if any(v.lower().startswith("pluripotency") for v in values):
            header_row = i
            break
    if header_row is None:
        raise SystemExit(f"no gene-set header row found in {path}")
    names = [str(v).split("\n")[0].strip() for v in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1:]
    sets: dict[str, list[str]] = {}
    for j, name in enumerate(names):
        if not name or name.lower() == "nan":
            continue
        column = body.iloc[:, j].dropna().astype(str).str.strip()
        sets[name] = [g for g in column if g and g.lower() != "nan"]
    return sets


def load_test(state: str, design: str) -> list[tuple[int, object, object]]:
    import anndata as ad

    out = []
    root = DESIGNS[design] / state
    for fold_dir in sorted(root.glob("fold_*")):
        path = fold_dir / f"gse283614_{state}_test.h5ad"
        if path.exists():
            out.append((int(fold_dir.name.split("_")[1]), ad.read_h5ad(path), path))
    return out


def decompose(adata, score: np.ndarray, strata: np.ndarray) -> dict:
    """Per-gene centered variance shares: guide / stratum-within-guide / residual."""
    import scipy.sparse as sp

    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float64, copy=False)
    X -= X.mean(axis=0, keepdims=True)
    total = float((X ** 2).sum())
    guides = adata.obs["guide_identity"].astype(str).to_numpy()
    between_guide = 0.0
    for g in np.unique(guides):
        mask = guides == g
        block = X[mask]
        between_guide += float(block.shape[0] * (block.mean(axis=0) ** 2).sum())
    within_guide_stratum = 0.0
    for g in np.unique(guides):
        gmask = guides == g
        guide_mean = X[gmask].mean(axis=0)
        for s in np.unique(strata):
            mask = gmask & (strata == s)
            if mask.sum() == 0:
                continue
            block = X[mask]
            within_guide_stratum += float(block.shape[0] * ((block.mean(axis=0) - guide_mean) ** 2).sum())
    residual = total - between_guide - within_guide_stratum
    return {
        "total_ss": total,
        "between_guide_share": between_guide / total if total else np.nan,
        "within_guide_between_stratum_share": within_guide_stratum / total if total else np.nan,
        "residual_share": residual / total if total else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=WS / "reports/assets/gse283614_de_vs_esc")
    parser.add_argument("--designs", nargs="+", default=["kd3", "dose_cv"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gene_sets = read_gene_sets(GENE_SET_WORKBOOK)
    print("gene sets:", {k: len(v) for k, v in gene_sets.items()})
    # The container rescorer needs the same state-score gene lists; export them
    # instead of re-parsing the workbook inside the image.
    pd.DataFrame(
        [(state, name, gene)
         for state, name in STATE_SCORE_SET.items()
         for gene in gene_sets[name]],
        columns=["state", "gene_set", "gene"],
    ).to_csv(args.output_dir / "state_score_genes.tsv", sep="\t", index=False)
    print(f"  wrote state_score_genes.tsv")

    decomp_rows, comp_rows = [], []
    for design in args.designs:
        for state in STATES:
            set_name = STATE_SCORE_SET[state]
            for fold, adata, path in load_test(state, design):
                genes = adata.var_names.astype(str)
                index = {g: i for i, g in enumerate(genes)}
                set_idx = [index[g] for g in gene_sets[set_name] if g in index]
                X = adata.X
                X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
                score = X[:, set_idx].mean(axis=1)
                extra = {}
                for set_name2 in EXTRA_SETS:
                    idx2 = [index[g] for g in gene_sets[set_name2] if g in index]
                    if idx2:
                        extra[set_name2] = X[:, idx2].mean(axis=1)
                edges = np.quantile(score, [1 / 3, 2 / 3])
                strata = np.digitize(score, edges)
                labels = np.array(["low", "mid", "high"])[strata]
                adata.obs["state_score"] = score
                adata.obs["stratum"] = labels
                for set_name2, values in extra.items():
                    adata.obs[f"score::{set_name2}"] = values
                dec = decompose(adata, score, strata)
                dec.update({"design": design, "state": state, "fold": fold,
                            "n_cells": int(adata.n_obs),
                            "n_guides": int(adata.obs["guide_identity"].nunique()),
                            "score_gene_set": set_name,
                            "n_score_genes": len(set_idx),
                            "mid_fraction": float((labels == "mid").mean()),
                            "score_sd": float(score.std())})
                decomp_rows.append(dec)

                for (guide, sub) in adata.obs.groupby("guide_identity", observed=True):
                    frac = (sub["stratum"].value_counts(normalize=True)
                            .reindex(["low", "mid", "high"]).fillna(0.0))
                    row = {
                        "design": design, "state": state, "fold": fold,
                        "guide": str(guide), "n_cells": int(len(sub)),
                        "mean_state_score": float(sub["state_score"].mean()),
                        "sd_state_score": float(sub["state_score"].std()),
                        "frac_low": float(frac["low"]), "frac_mid": float(frac["mid"]),
                        "frac_high": float(frac["high"]),
                    }
                    for set_name2 in extra:
                        row[f"mean_score::{set_name2}"] = float(
                            sub[f"score::{set_name2}"].mean())
                    comp_rows.append(row)
    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(args.output_dir / "heterogeneity_decomposition.csv", index=False)
    print(f"  wrote heterogeneity_decomposition.csv ({len(decomp)} rows)")

    comp = pd.DataFrame(comp_rows)
    dose = pd.read_csv(DIFFAE / "runs/biological_validation/gse283614_observed/"
                              "gse283614_tf_dosage.tsv", sep="\t")
    comp = comp.merge(dose[["guide_identity", "dose_target", "knockdown_fraction"]],
                      left_on="guide", right_on="guide_identity", how="left").drop(
        columns=["guide_identity"])
    # pooled per guide across folds (each guide is held out exactly once per design)
    extra_cols = [c for c in comp.columns if c.startswith("mean_score::")]
    agg = {"n_cells": "sum", "mean_state_score": "mean", "sd_state_score": "mean",
           "frac_low": "mean", "frac_mid": "mean", "frac_high": "mean",
           "knockdown_fraction": "first", "dose_target": "first"}
    agg.update({c: "mean" for c in extra_cols})
    pooled = comp.groupby(["design", "state", "guide"], as_index=False).agg(agg)
    pooled.to_csv(args.output_dir / "stratum_composition.csv", index=False)
    print(f"  wrote stratum_composition.csv ({len(pooled)} rows)")

    # does dose move the composition?  and does composition track the metric damage?
    corr_rows = []
    for (design, state), block in pooled.groupby(["design", "state"]):
        for x in ["knockdown_fraction", "n_cells"]:
            for y in ["frac_low", "frac_mid", "frac_high", "mean_state_score", "sd_state_score", "n_cells"]:
                if x == y:
                    continue
                pair = block[[x, y]].dropna()
                if len(pair) < 6 or pair[x].nunique() < 3:
                    continue
                rho, p = stats.spearmanr(pair[x], pair[y])
                corr_rows.append({"design": design, "state": state, "x": x, "y": y,
                                  "spearman_rho": float(rho), "p_value": float(p),
                                  "n": int(len(pair))})
        analysis = WS / (f".scperturbench/results/gse283614_activity_ablation{'_kd3' if design == 'kd3' else ''}"
                         f"/gse283614/{state}/ablation_analysis/ablation_per_condition.csv")
        if analysis.exists():
            frame = pd.read_csv(analysis)
            for (metric, budget), mblock in frame.groupby(["metric", "deg_budget"], sort=False):
                if (metric, str(budget)) not in [(m, b) for m, b in PRIMARY]:
                    continue
                merged = block.merge(
                    mblock[["condition", "paired_delta_on_minus_off", "higher_is_better"]],
                    left_on="guide", right_on="condition", how="inner").dropna(
                    subset=["paired_delta_on_minus_off"])
                if len(merged) < 6:
                    continue
                higher = bool(merged["higher_is_better"].iloc[0])
                merged["damage"] = merged["paired_delta_on_minus_off"] * (-1 if higher else 1)
                for x in (["frac_high", "frac_low", "mean_state_score", "knockdown_fraction"]
                          + extra_cols):
                    pair = merged[[x, "damage"]].dropna()
                    if len(pair) < 6 or pair[x].nunique() < 3:
                        continue
                    rho, p = stats.spearmanr(pair[x], pair["damage"])
                    corr_rows.append({"design": design, "state": state, "x": x,
                                      "y": f"damage::{metric}@{budget}",
                                      "spearman_rho": float(rho), "p_value": float(p),
                                      "n": int(len(pair))})
    corr = pd.DataFrame(corr_rows)
    if not corr.empty:
        corr["q_bh"] = np.nan
        for (design, state), block in corr.groupby(["design", "state"]):
            corr.loc[block.index, "q_bh"] = _bh(block["p_value"].to_numpy(dtype=float))
    corr.to_csv(args.output_dir / "stratum_composition_corr.csv", index=False)
    print(f"  wrote stratum_composition_corr.csv ({len(corr)} rows)")

    print()
    print(decomp[["design", "state", "fold", "n_cells", "n_guides",
                  "between_guide_share", "within_guide_between_stratum_share",
                  "residual_share", "mid_fraction"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def _bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.argsort(p[finite])
    ranked = p[finite][idx]
    n = ranked.size
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[idx] = np.clip(adjusted, 0.0, 1.0)
    q[finite] = out
    return q


if __name__ == "__main__":
    raise SystemExit(main())
