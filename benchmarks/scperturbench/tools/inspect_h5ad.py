"""Inspect an .h5ad file: shape, obs columns, layers, X stats.

Host-side helper; run inside the container where anndata is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import anndata as ad
from scipy import sparse


def describe(path: Path, sample: int) -> dict:
    adata = ad.read_h5ad(path, backed="r")
    info: dict = {
        "path": str(path),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": list(map(str, adata.obs.columns)),
        "var_columns": list(map(str, adata.var.columns)),
        "layers": list(map(str, adata.layers.keys())),
        "obsm": list(map(str, adata.obsm.keys())),
        "uns": list(map(str, adata.uns.keys())),
        "first_genes": [str(g) for g in adata.var_names[:8]],
    }
    if "gene_name" in adata.var.columns:
        info["gene_name_example"] = [str(g) for g in adata.var["gene_name"][:5]]
    for column in ("perturbation_label", "condition", "guide", "Group", "target_gene", "n_genes"):
        if column in adata.obs.columns:
            values = adata.obs[column].astype(str)
            info[f"obs_{column}_nunique"] = int(values.nunique())
            info[f"obs_{column}_head"] = list(values.value_counts().head(8).items())
    # X statistics on the first `sample` rows
    x = adata.X[:sample]
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    info["x_sample_rows"] = int(x.shape[0])
    info["x_mean"] = float(x.mean())
    info["x_std"] = float(x.std())
    info["x_max"] = float(x.max())
    info["x_min"] = float(x.min())
    info["x_frac_zero"] = float((x == 0).mean())
    info["x_is_integer_like"] = bool(np.allclose(x, np.round(x)))
    adata.file.close()
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--sample", type=int, default=200)
    args = parser.parse_args()
    for path in args.paths:
        try:
            print(json.dumps(describe(path, args.sample), indent=2, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"path": str(path), "error": repr(exc)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
