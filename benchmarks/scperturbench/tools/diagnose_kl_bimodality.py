"""Explain the bimodal sym-KL in the GSE132080 activity ablation.

Recomputes the per-gene symmetric-KL contributions used by ``sym_kldiv_with_floor`` for each
evaluated condition and reports which genes dominate.  Hypothesis: the metric is dominated by genes
whose *stimulated* (ground-truth) standard deviation is ~0, which drives the
``(x_mean - y_mean)^2 / (2 y_std^2)`` term through the ``epsilon = 1e-8`` floor and leaves the raw
value around 2**24-2**30 for every condition -- i.e. the "high cluster" is a floor artifact, not a
biological difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/root/perturbench/.scperturbench/results/gse132080_activity_ablation/gse132080"),
    )
    parser.add_argument("--arm", default="activity_on_200ep")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("/root/perturbench/.scperturbench/inputs/gse132080_squidiff_repro/squidiff_test.h5ad"),
    )
    parser.add_argument("--deg-table", type=Path, default=None)
    parser.add_argument("--top", type=int, default=4)
    return parser.parse_args()


def per_gene_sym_kl(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    x_std = x.std(axis=0) + epsilon
    y_std = y.std(axis=0) + epsilon
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    kl = np.log(y_std / x_std) + (x_std**2 + (x_mean - y_mean) ** 2) / (2 * y_std**2) - 0.5
    klr = np.log(x_std / y_std) + (y_std**2 + (y_mean - x_mean) ** 2) / (2 * x_std**2) - 0.5
    return kl + klr


def main() -> int:
    args = parse_args()
    result_path = args.results_root / args.arm / f"seed{args.seed}" / "result.h5ad"
    adata = ad.read_h5ad(result_path)
    genes = np.asarray(adata.var_names.astype(str))

    if args.deg_table is None:
        args.deg_table = args.results_root.parent.parent.parent / "inputs/gse132080_bench/deg_table.json"
    deg_lookup = {}
    if Path(args.deg_table).exists():
        payload = json.loads(Path(args.deg_table).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            # The prepared table is {"recipe":..., "dataset":..., "genes": {condition: {...}}};
            # load_deg_table() unwraps the outer "genes" key, so mirror that here.
            deg_lookup = payload.get("genes", payload) if "genes" in payload else payload
        print(f"deg table: {args.deg_table.name} ({len(deg_lookup)} conditions)")

    def deg_gene_names(condition: str, limit: int | None = None) -> list[str]:
        """Mirror deg_list(): the ``order`` field is already sorted by |t score|."""
        entry = deg_lookup.get(condition) if isinstance(deg_lookup, dict) else None
        if not isinstance(entry, dict):
            return []
        order = [str(g) for g in entry.get("order", [])]
        return order if limit is None else order[:limit]

    rows = []
    print(f"\n{'condition':<10} {'raw_all':>10} {'raw_deg':>11} {'log2+1':>8} {'n_deg':>6} "
          f"{'deg_y0':>7} {'deg_x0':>7} {'top_deg_gene':<12} {'share':>7} {'y_std':>8} {'|ymean-xmean|':>13}")
    for condition in sorted(set(adata.obs["perturbation"].astype(str))):
        if condition == "control":
            continue
        block = adata[adata.obs["perturbation"].astype(str) == condition]
        x_all = np.asarray(block[block.obs["Expcategory"] == "imputed"].X, dtype=np.float64)
        y_all = np.asarray(block[block.obs["Expcategory"] == "stimulated"].X, dtype=np.float64)
        if x_all.shape[0] == 0 or y_all.shape[0] == 0:
            continue

        deg_idx = np.arange(len(genes))
        names = deg_gene_names(condition, limit=100)
        if names:
            keep = set(names)
            deg_idx = np.array([i for i, g in enumerate(genes) if g in keep], dtype=int)
        x = x_all[:, deg_idx]
        y = y_all[:, deg_idx]

        contrib = per_gene_sym_kl(x, y)
        raw_deg = float(contrib.mean())
        raw_all = float(per_gene_sym_kl(x_all, y_all).mean())
        order = np.argsort(contrib)[::-1]
        top = order[0]
        y_std = float(y[:, top].std())
        mean_gap = abs(float(y[:, top].mean()) - float(x[:, top].mean()))

        rows.append(
            {
                "condition": condition,
                "raw_all": raw_all,
                "raw_deg": raw_deg,
                "log2_kldiv": float(np.log2(raw_deg + 1)),
                "n_deg": int(len(deg_idx)),
                "n_deg_y_std_lt_1e-6": int((y.std(axis=0) < 1e-6).sum()),
                "n_deg_x_std_lt_1e-6": int((x.std(axis=0) < 1e-6).sum()),
                "top_deg_gene": str(genes[deg_idx[top]]) if len(deg_idx) else "",
                "top_share": float(contrib[top] / contrib.sum()) if contrib.sum() else float("nan"),
                "top_y_std": y_std,
                "top_mean_gap": mean_gap,
            }
        )
        print(f"{condition:<10} {raw_all:>10.4g} {raw_deg:>11.4g} {np.log2(raw_deg+1):>8.3f} "
              f"{len(deg_idx):>6} {int((y.std(axis=0) < 1e-6).sum()):>7} {int((x.std(axis=0) < 1e-6).sum()):>7} "
              f"{str(genes[deg_idx[top]]) if len(deg_idx) else '':<12} "
              f"{float(contrib[top]/contrib.sum()) if contrib.sum() else float('nan'):>6.1%} "
              f"{y_std:>8.3g} {mean_gap:>13.4g}")

    frame = pd.DataFrame(rows).sort_values("log2_kldiv")
    print("\n=== sorted by the reported metric (DEG-100) ===")
    print(frame[["condition", "log2_kldiv", "raw_deg", "raw_all", "n_deg_y_std_lt_1e-6",
                 "n_deg_x_std_lt_1e-6", "top_deg_gene"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    low = frame[frame["log2_kldiv"] < 10]
    high = frame[frame["log2_kldiv"] >= 10]
    print(f"\nlow cluster  (<10): n={len(low):>2}  {list(low['condition'])}")
    print(f"high cluster (>=10): n={len(high):>2}  {list(high['condition'])}")
    if len(low) and len(high):
        print(f"\nmean genes with zero y-variance in DEG set: low={low['n_deg_y_std_lt_1e-6'].mean():.1f} "
              f"high={high['n_deg_y_std_lt_1e-6'].mean():.1f}")
        print(f"mean genes with zero x-variance in DEG set: low={low['n_deg_x_std_lt_1e-6'].mean():.1f} "
              f"high={high['n_deg_x_std_lt_1e-6'].mean():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
