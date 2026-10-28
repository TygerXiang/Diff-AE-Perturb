"""Is the sym-KL bimodality real, or a blow-up artifact from single zero-variance genes?

For each evaluated condition, recompute the symmetric KL restricted to the same DEG-100 subset the
scorer uses, then compare the two visual clusters on:

  * the reported (literal) value, which the epsilon=1e-8 floor lets a single gene dominate,
  * the floored diagnostic companion (std floor 0.1), which is immune to that blow-up,
  * and the per-gene contribution of the single worst gene.

If the "high" cluster differs only on the literal value, the bimodality is an artifact of the
zero-variance term and says nothing about the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path("/root/perturbench/.scperturbench/results/gse132080_activity_ablation/gse132080")
DEG = Path("/root/perturbench/.scperturbench/inputs/gse132080_bench/deg_table.json")
EPS = 1e-8
FLOOR = 0.1


def sym_kl_per_gene(x: np.ndarray, y: np.ndarray, *, std_floor: float = 0.0) -> np.ndarray:
    x_std = np.maximum(x.std(axis=0) + EPS, std_floor) if std_floor else x.std(axis=0) + EPS
    y_std = np.maximum(y.std(axis=0) + EPS, std_floor) if std_floor else y.std(axis=0) + EPS
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    kl = np.log(y_std / x_std) + (x_std**2 + (x_mean - y_mean) ** 2) / (2 * y_std**2) - 0.5
    klr = np.log(x_std / y_std) + (y_std**2 + (y_mean - x_mean) ** 2) / (2 * x_std**2) - 0.5
    return kl + klr


def main() -> int:
    deg = json.loads(DEG.read_text(encoding="utf-8"))["genes"]
    frames = {}
    for arm in ("activity_on_200ep", "activity_off_200ep"):
        adata = ad.read_h5ad(ROOT / arm / "seed7" / "result.h5ad")
        rows = []
        for condition in sorted(set(adata.obs["perturbation"].astype(str))):
            if condition == "control":
                continue
            genes = [g for g in deg[condition]["order"][:100]]
            block = adata[adata.obs["perturbation"].astype(str) == condition]
            imp = block[block.obs["Expcategory"] == "imputed"]
            sti = block[block.obs["Expcategory"] == "stimulated"]
            keep = [g for g in genes if g in adata.var_names]
            x = np.asarray(imp[:, keep].X, dtype=np.float64)
            y = np.asarray(sti[:, keep].X, dtype=np.float64)

            literal = sym_kl_per_gene(x, y)
            floored = sym_kl_per_gene(x, y, std_floor=FLOOR)
            top = int(np.argmax(literal))
            rows.append(
                {
                    "condition": condition,
                    "n_cells": int(x.shape[0]),
                    "reported_log2": float(np.log2(literal.mean() + 1)),
                    "floored_log2": float(np.log2(floored.mean() + 1)),
                    "worst_gene": keep[top],
                    "worst_share_pct": float(100 * literal[top] / literal.sum()),
                    "worst_gene_y_std": float(y[:, top].std()),
                    "n_genes_zero_y_std": int((y.std(axis=0) < 1e-6).sum()),
                }
            )
        frames[arm] = pd.DataFrame(rows).set_index("condition")

    on, off = frames["activity_on_200ep"], frames["activity_off_200ep"]
    merged = on.join(off, lsuffix="_on", rsuffix="_off")
    merged["cluster"] = np.where(merged["reported_log2_on"] >= 10, "high", "low")

    print("=== per condition: literal vs floored sym-KL (activity_on arm) ===")
    print(
        on[["n_cells", "reported_log2", "floored_log2", "worst_gene", "worst_share_pct",
            "worst_gene_y_std", "n_genes_zero_y_std"]]
        .sort_values("reported_log2")
        .to_string(float_format=lambda v: f"{v:.4g}")
    )

    print("\n=== cluster comparison (literal vs floored) ===")
    summary = merged.groupby("cluster")[
        ["reported_log2_on", "reported_log2_off", "floored_log2_on", "floored_log2_off",
         "n_cells_on", "worst_share_pct_on"]
    ].mean()
    print(summary.to_string(float_format=lambda v: f"{v:.4g}"))

    print("\n=== does the cluster split explain the between-arm difference? ===")
    merged["delta_reported"] = merged["reported_log2_on"] - merged["reported_log2_off"]
    merged["delta_floored"] = merged["floored_log2_on"] - merged["floored_log2_off"]
    print(merged.groupby("cluster")[["delta_reported", "delta_floored"]].agg(["mean", "std"]).to_string(
        float_format=lambda v: f"{v:.4g}"
    ))
    print(f"\noverall mean literal delta : {merged['delta_reported'].mean():+.4f}")
    print(f"overall mean floored delta : {merged['delta_floored'].mean():+.4f}")
    print(f"conditions in the high cluster: {list(merged[merged['cluster'] == 'high'].index)}")
    print(f"their worst-gene share of the literal value: "
          f"{merged[merged['cluster'] == 'high']['worst_share_pct_on'].mean():.1f}% (mean)")
    print(f"low-cluster worst-gene share: "
          f"{merged[merged['cluster'] == 'low']['worst_share_pct_on'].mean():.1f}% (mean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
