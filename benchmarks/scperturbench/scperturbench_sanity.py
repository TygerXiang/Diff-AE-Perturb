"""Quick sanity check of one result.h5ad: value ranges, per-block means, and a
pseudobulk comparison against the ground truth (run inside the container)."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


def dense(x) -> np.ndarray:
    return np.asarray(x.toarray()) if sparse.issparse(x) else np.asarray(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-h5ad", type=Path, required=True)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.result_h5ad)
    X = dense(adata.X)
    print(f"shape={adata.shape} obs cols={list(adata.obs.columns)}")
    for category in ["control", "stimulated", "imputed"]:
        mask = adata.obs["Expcategory"].astype(str) == category
        block = X[mask]
        print(
            f"{category:11s} n={block.shape[0]:5d} min={block.min():.4f} max={block.max():.4f} "
            f"mean={block.mean():.4f} nan={int(np.isnan(block).sum())} zero_frac={(block == 0).mean():.3f}"
        )

    conditions = sorted(set(adata.obs.loc[adata.obs["Expcategory"].astype(str) == "imputed", "perturbation"].astype(str)))
    chosen = [args.condition] if args.condition else conditions[:3]
    label = adata.obs["perturbation"].astype(str)
    for condition in chosen:
        true_mask = (label == condition) & (adata.obs["Expcategory"].astype(str) == "stimulated")
        pred_mask = (label == condition) & (adata.obs["Expcategory"].astype(str) == "imputed")
        true_mean = X[true_mask].mean(axis=0)
        pred_mean = X[pred_mask].mean(axis=0)
        r = float(np.corrcoef(true_mean, pred_mean)[0, 1]) if true_mean.std() > 0 and pred_mean.std() > 0 else float("nan")
        print(
            f"{condition:8s} n_true={int(true_mask.sum()):4d} n_pred={int(pred_mask.sum()):4d} "
            f"pearson_pseudobulk={r:.4f} mse_pseudobulk={float(((true_mean - pred_mean) ** 2).mean()):.6f} "
            f"true_mean_avg={true_mean.mean():.4f} pred_mean_avg={pred_mean.mean():.4f}"
        )
        order = np.argsort(-np.abs(true_mean - true_mean.mean()))[: args.top]
        genes = [str(adata.var_names[i]) for i in order]
        print(f"  top-variance genes: {list(zip(genes, np.round(true_mean[order], 3), np.round(pred_mean[order], 3)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
