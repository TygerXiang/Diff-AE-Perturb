"""Compare the project's GSE132080 preprocessing recipe against Squidiff's own prep_* recipe.

Both are run from the raw GSE132080 10x counts, each with its own documented QC thresholds:

  project  : preprocess_gse132080.py  (total_counts>=1000, 200<=n_genes<=10000, pct_mt<20,
             min_cells>=3, drop MT-/RPS/RPL, normalize_total(1e4), log1p,
             highly_variable_genes(n_top_genes=1000, flavor='seurat') | target genes)
  squidiff : prep_drug_screen_data.ipynb (filter_cells(min_genes=300), filter_genes(min_cells=3),
             drop MT-/RP- genes, normalize_total(1e4), log1p,
             highly_variable_genes(flavor='seurat', n_top_genes=1000) + forced genes of interest)

The point is to measure how much of the project's 1019-gene panel is actually reproduced by
Squidiff's own recipe rather than by the project's own choices.

Run inside the container (env `squidiff`).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread


def read_tsv_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return [line.rstrip("\n").split("\t") for line in handle]


def load_raw(data_dir: Path) -> ad.AnnData:
    matrix = mmread(data_dir / "matrix.mtx.gz").tocsr().transpose().tocsr()
    barcodes = [row[0] for row in read_tsv_rows(data_dir / "barcodes.tsv.gz")]
    gene_rows = read_tsv_rows(data_dir / "genes.tsv.gz")
    names = [row[1] if len(row) > 1 else row[0] for row in gene_rows]
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=pd.Index(barcodes, name="barcode")),
        var=pd.DataFrame(index=pd.Index(names, name="gene")),
    )
    adata.var_names_make_unique()
    return adata


def clean_label(value) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if "neg_ctrl" in lowered or "non-targeting" in lowered or "non_targeting" in lowered:
        return "control"
    return text.split("_", 1)[0]


def normalise(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def hvg_panel(adata: ad.AnnData, n_top: int, targets: list[str], force: bool):
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top, adata.n_vars), flavor="seurat", subset=False)
    hvg = [str(g) for g, flag in zip(adata.var_names, adata.var["highly_variable"]) if flag]
    if force:
        present = [g for g in targets if g in set(map(str, adata.var_names))]
        panel = sorted(set(hvg) | set(present))
    else:
        panel = sorted(hvg)
    return panel, hvg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--benchmark-genes", type=Path, required=True)
    parser.add_argument("--evaluated-conditions", type=Path, required=True, help="conditions.json")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    report: dict = {}

    # ---------------------------------------------------------------- raw
    raw = load_raw(args.raw_dir)
    report["raw"] = {"cells": int(raw.n_obs), "genes": int(raw.n_vars)}

    meta = pd.read_csv(args.metadata, index_col=0)
    meta.index = meta.index.astype(str)
    keep = [b for b in raw.obs_names if b in set(meta.index)]
    aligned = raw[keep].copy()
    for column in meta.columns:
        aligned.obs[column] = meta.loc[aligned.obs_names, column].to_numpy()
    aligned.obs["perturbation_label"] = [clean_label(v) for v in aligned.obs["guide_identity"]]
    targets = sorted(
        {str(v) for v in aligned.obs["perturbation_label"] if str(v) not in {"control", "*", ""}}
    )
    report["after_metadata_align"] = {"cells": int(aligned.n_obs), "target_genes_present": len(targets)}

    evaluated = json.loads(Path(args.evaluated_conditions).read_text())["evaluated_conditions"]
    benchmark = [str(g) for g in json.loads(Path(args.benchmark_genes).read_text())["genes"]]

    # ---------------------------------------------------------------- project recipe
    a = aligned.copy()
    if "good_coverage" in a.obs:
        a = a[a.obs["good_coverage"].astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})].copy()
    if "number_of_cells" in a.obs:
        a = a[pd.to_numeric(a.obs["number_of_cells"], errors="coerce").fillna(np.inf) <= 1].copy()
    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    mask = (
        (a.obs["total_counts"] >= 1000)
        & (a.obs["n_genes_by_counts"] >= 200)
        & (a.obs["n_genes_by_counts"] <= 10000)
        & (a.obs["pct_counts_mt"] < 20)
    )
    a = a[mask.to_numpy()].copy()
    x = a.X.tocsr()
    gene_cells = np.asarray((x > 0).sum(axis=0)).reshape(-1)
    non_mt_ribo = np.asarray(~a.var_names.str.upper().str.startswith(("MT-", "RPS", "RPL")))
    a = a[:, ((gene_cells >= 3) & non_mt_ribo)].copy()
    a.var_names_make_unique()
    project_cells = int(a.n_obs)
    a = normalise(a)
    panel_project, hvg_project = hvg_panel(a, 1000, targets, force=True)
    report["project_recipe"] = {
        "cells": project_cells,
        "genes_before_hvg": int(a.n_vars),
        "n_hvg": len(hvg_project),
        "panel_size": len(panel_project),
        "panel_equals_benchmark": sorted(panel_project) == sorted(benchmark),
    }

    # ---------------------------------------------------------------- squidiff recipe
    b = aligned.copy()
    sc.pp.filter_cells(b, min_genes=300)
    sc.pp.filter_genes(b, min_cells=3)
    b.var["mt"] = np.logical_or(
        b.var_names.str.upper().str.startswith("MT-"), b.var_names.str.upper().str.startswith("RP-")
    )
    b = b[:, np.asarray(~b.var["mt"])].copy()
    squidiff_cells = int(b.n_obs)
    b = normalise(b)
    panel_squidiff, hvg_squidiff = hvg_panel(b, 1000, targets, force=True)
    panel_squidiff_hvg_only, _ = hvg_panel(b, 1000, targets, force=False)
    report["squidiff_recipe"] = {
        "cells": squidiff_cells,
        "genes_before_hvg": int(b.n_vars),
        "n_hvg": len(hvg_squidiff),
        "panel_size": len(panel_squidiff),
        "panel_hvg_only_size": len(panel_squidiff_hvg_only),
    }

    # ---------------------------------------------------------------- comparison
    shared = sorted(set(panel_project) & set(panel_squidiff))
    report["comparison"] = {
        "project_panel": len(panel_project),
        "squidiff_panel": len(panel_squidiff),
        "identical": sorted(panel_project) == sorted(panel_squidiff),
        "shared": len(shared),
        "only_in_project": sorted(set(panel_project) - set(panel_squidiff)),
        "only_in_squidiff": sorted(set(panel_squidiff) - set(panel_project)),
        "hvg_vs_hvg_shared": len(set(hvg_project) & set(hvg_squidiff)),
        "evaluated_conditions": evaluated,
        "evaluated_conditions_in_project_panel": [c for c in evaluated if c in set(panel_project)],
        "evaluated_conditions_in_squidiff_panel": [c for c in evaluated if c in set(panel_squidiff)],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
