from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-style preprocessing for GSE132080: normalize_total -> log1p -> /10 -> HVG."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "GSE132080",
        help="Directory with matrix.mtx.gz, barcodes.tsv.gz, genes.tsv.gz, and metadata csv.gz.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=REPO_ROOT / "datasets" / "GSE132080" / "GSE132080_cell_identities.csv.gz",
    )
    parser.add_argument("--barcode-col", default="cell_barcode")
    parser.add_argument("--perturbation-col", default="guide_identity")
    parser.add_argument("--clean-perturbation-col", default="perturbation_label")
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "gse132080_paper_style_hvg2000.h5ad",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "gse132080_paper_style_hvg2000_summary.json",
    )
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--hvg-n-top", type=int, default=2000)
    parser.add_argument("--hvg-flavor", default="seurat_v3")
    parser.add_argument(
        "--keep-good-coverage-only",
        action="store_true",
        help="Optionally keep only good_coverage=true cells from metadata.",
    )
    return parser.parse_args()


def _read_tsv_first_col(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        return [line.rstrip("\n").split("\t")[0] for line in f]


def _read_gene_names(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    return [r[1] if len(r) > 1 else r[0] for r in rows]


def _clean_label(x: object) -> str:
    s = str(x).strip()
    lo = s.lower()
    if "neg_ctrl" in lo or "non-targeting" in lo or "non_targeting" in lo:
        return "control"
    return s


def main() -> None:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required. Install it in your conda env first.") from exc

    args = parse_args()
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    matrix_path = args.data_dir / "matrix.mtx.gz"
    barcodes_path = args.data_dir / "barcodes.tsv.gz"
    genes_path = args.data_dir / "genes.tsv.gz"

    X = mmread(matrix_path).tocsr().transpose().tocsr()
    barcodes = _read_tsv_first_col(barcodes_path)
    genes = _read_gene_names(genes_path)

    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()

    meta = pd.read_csv(args.metadata_path)
    if args.barcode_col not in meta.columns:
        raise KeyError(f"Missing metadata barcode column: {args.barcode_col}")
    if args.perturbation_col not in meta.columns:
        raise KeyError(f"Missing metadata perturbation column: {args.perturbation_col}")

    meta = meta.drop_duplicates(subset=[args.barcode_col]).set_index(args.barcode_col)
    keep = [bc for bc in adata.obs_names if bc in meta.index]
    adata = adata[keep].copy()
    adata.obs = adata.obs.join(meta, how="left")

    if args.keep_good_coverage_only and "good_coverage" in adata.obs.columns:
        good = adata.obs["good_coverage"].astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})
        adata = adata[good.to_numpy()].copy()

    adata.obs[args.clean_perturbation_col] = adata.obs[args.perturbation_col].map(_clean_label).astype(str)

    # A.2.1: normalize total -> log1p -> scale by 10 to [0, 1) empirical range.
    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)
    adata.X = adata.X / 10.0

    # A.2.2: select top 2000 HVGs (or all if dataset has fewer genes).
    n_top = min(args.hvg_n_top, adata.n_vars)
    if args.hvg_flavor == "seurat_v3":
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor=args.hvg_flavor, layer="raw_counts")
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor=args.hvg_flavor)
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()

    adata.uns["paper_style_preprocessing"] = {
        "steps": [
            "normalize_total",
            "log1p",
            "divide_by_10",
            f"highly_variable_genes_top_{n_top}",
        ],
        "target_sum": args.target_sum,
        "hvg_flavor": args.hvg_flavor,
        "hvg_n_top_requested": args.hvg_n_top,
        "hvg_n_top_used": n_top,
    }

    adata.write_h5ad(args.output_h5ad)

    summary = {
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "x_min": float(np.min(adata.X)),
        "x_max": float(np.max(adata.X)),
        "x_mean": float(np.mean(adata.X)),
        "perturbation_counts": adata.obs[args.clean_perturbation_col].value_counts().to_dict(),
        "output_h5ad": str(args.output_h5ad),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {args.output_h5ad}")
    print(f"summary: {args.summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

