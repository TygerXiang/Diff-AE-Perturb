from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.data import read_metadata_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GSE132080 into a clean AnnData object.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "GSE132080",
        help="Directory containing matrix.mtx.gz, barcodes.tsv.gz, and genes.tsv.gz.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="CSV/TSV metadata table with cell_barcode and guide_identity columns.",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "gse132080_preprocessed.h5ad",
        help="Path to the prepared AnnData file.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "gse132080_preprocessing_summary.json",
        help="Optional JSON summary written alongside the .h5ad file.",
    )
    parser.add_argument("--barcode-col", default="cell_barcode")
    parser.add_argument("--perturbation-col", default="guide_identity")
    parser.add_argument("--clean-perturbation-col", default="perturbation_label")
    parser.add_argument("--min-counts", type=float, default=1000)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=10000)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--n-top-genes", type=int, default=1000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--skip-target-gene-retention", action="store_true")
    return parser.parse_args()


def read_tsv_rows(path: Path) -> list[list[str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return [line.rstrip("\n").split("\t") for line in handle]


def load_raw_adata(data_dir: Path) -> ad.AnnData:
    matrix_path = next((data_dir / name for name in ("matrix.mtx", "matrix.mtx.gz") if (data_dir / name).exists()), None)
    barcode_path = next((data_dir / name for name in ("barcodes.tsv", "barcodes.tsv.gz") if (data_dir / name).exists()), None)
    gene_path = next((data_dir / name for name in ("genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz") if (data_dir / name).exists()), None)
    if matrix_path is None or barcode_path is None or gene_path is None:
        raise FileNotFoundError("Expected matrix.mtx(.gz), barcodes.tsv(.gz), and genes.tsv(.gz) or features.tsv(.gz) in data_dir")

    matrix = mmread(matrix_path).tocsr().transpose().tocsr()
    barcodes = [row[0] for row in read_tsv_rows(barcode_path)]
    gene_rows = read_tsv_rows(gene_path)
    gene_names = [row[1] if len(row) > 1 else row[0] for row in gene_rows]

    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def attach_metadata(adata: ad.AnnData, metadata_path: Path, barcode_col: str) -> tuple[ad.AnnData, int]:
    metadata_rows = read_metadata_table(metadata_path)
    if not metadata_rows:
        raise ValueError(f"No rows found in metadata table: {metadata_path}")
    if barcode_col not in metadata_rows[0]:
        raise KeyError(f"Column '{barcode_col}' not found in metadata table")

    by_barcode = {row[barcode_col]: row for row in metadata_rows if row.get(barcode_col)}
    keep_barcodes = [barcode for barcode in adata.obs_names if barcode in by_barcode]
    dropped = len(adata) - len(keep_barcodes)
    adata = adata[keep_barcodes].copy()

    for key in metadata_rows[0].keys():
        adata.obs[key] = [by_barcode[barcode].get(key, "") for barcode in adata.obs_names]

    return adata, dropped


def clean_perturbation_label(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if "neg_ctrl" in lowered or "non-targeting" in lowered or "non_targeting" in lowered:
        return "control"
    return text.split("_", 1)[0]


def to_bool_series(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def as_1d(array_like: object) -> np.ndarray:
    return np.asarray(array_like).reshape(-1)


def main() -> None:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("This script requires scanpy. Please create a conda environment with scanpy installed.") from exc

    args = parse_args()
    metadata_path = args.metadata_path or (args.data_dir / "GSE132080_cell_identities.csv.gz")
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    adata = load_raw_adata(args.data_dir)
    print(f"loaded raw matrix: {adata.n_obs} cells x {adata.n_vars} genes")
    raw_cells = len(adata)
    adata, metadata_dropped = attach_metadata(adata, metadata_path, args.barcode_col)
    print(f"after metadata alignment: {adata.n_obs} cells")

    adata.obs[args.clean_perturbation_col] = [
        clean_perturbation_label(value) for value in adata.obs[args.perturbation_col]
    ]
    print("cleaned perturbation labels")

    before_coverage_filter = len(adata)
    if "good_coverage" in adata.obs:
        adata = adata[to_bool_series(adata.obs["good_coverage"])].copy()
    after_coverage_filter = len(adata)
    print(f"after good_coverage filter: {after_coverage_filter} cells")

    before_single_cell_filter = len(adata)
    if "number_of_cells" in adata.obs:
        adata = adata[pd.to_numeric(adata.obs["number_of_cells"], errors="coerce").fillna(np.inf) <= 1].copy()
    after_single_cell_filter = len(adata)
    print(f"after number_of_cells filter: {after_single_cell_filter} cells")

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    cell_mask = (
        (adata.obs["total_counts"] >= args.min_counts)
        & (adata.obs["n_genes_by_counts"] >= args.min_genes)
        & (adata.obs["n_genes_by_counts"] <= args.max_genes)
        & (adata.obs["pct_counts_mt"] < args.max_pct_mt)
    )
    adata = adata[cell_mask.to_numpy()].copy()
    print(f"after cell QC: {adata.n_obs} cells")

    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    gene_cells = as_1d((X > 0).sum(axis=0))
    # A gene that is itself perturbed must survive every gene filter, otherwise the model can
    # never see the perturbation it is asked to predict.  Computing the protected set here (and
    # not after the ribosomal filter) is what keeps RPS14/RPS15/RPS18/RPL9 in the panel; the old
    # order dropped them and the later target retention could not put them back.
    protected_targets = np.asarray(
        adata.var_names.isin(
            sorted(
                {
                    str(v)
                    for v in adata.obs[args.clean_perturbation_col]
                    if str(v) not in {"control", "*", ""}
                }
            )
        )
    )
    non_mt_ribo = np.asarray(~adata.var_names.str.upper().str.startswith(("MT-", "RPS", "RPL")))
    gene_mask = (gene_cells >= args.min_cells_per_gene) & (non_mt_ribo | protected_targets)
    adata = adata[:, gene_mask].copy()
    print(
        f"after gene QC: {adata.n_obs} cells x {adata.n_vars} genes "
        f"({int(protected_targets.sum())} perturbation target genes exempted from the MT/ribosomal filter)"
    )
    adata.var_names_make_unique()

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)
    print("normalized and log-transformed")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(args.n_top_genes, adata.n_vars),
        flavor=args.hvg_flavor,
        subset=False,
    )
    hvg_mask = adata.var["highly_variable"].to_numpy().astype(bool)
    n_hvg_selected = int(hvg_mask.sum())

    if not args.skip_target_gene_retention:
        target_genes = set(
            adata.obs.loc[adata.obs[args.clean_perturbation_col] != "control", args.clean_perturbation_col]
        )
        keep_targets = adata.var_names.isin(target_genes)
        target_mask = keep_targets.to_numpy() if hasattr(keep_targets, "to_numpy") else np.asarray(keep_targets)
        n_target_genes = int(target_mask.sum())
        n_target_overlap = int((hvg_mask & target_mask).sum())
        selection_mask = hvg_mask | target_mask
        adata.var["highly_variable"] = selection_mask
        print(f"selected HVGs: {n_hvg_selected}")
        print(f"target genes present: {n_target_genes}")
        print(f"target/HVG overlap: {n_target_overlap}")
        print(f"retained target genes: {n_target_genes}")
    else:
        selection_mask = hvg_mask
        n_target_genes = 0
        n_target_overlap = 0
        print(f"selected HVGs: {n_hvg_selected}")
        print("target gene retention disabled")

    adata = adata[:, selection_mask].copy()
    print(f"after HVG selection: {adata.n_obs} cells x {adata.n_vars} genes")
    adata.uns["preprocessing_params"] = {
        "data_dir": str(args.data_dir),
        "metadata_path": str(metadata_path),
        "barcode_col": args.barcode_col,
        "perturbation_col": args.perturbation_col,
        "clean_perturbation_col": args.clean_perturbation_col,
        "min_counts": args.min_counts,
        "min_genes": args.min_genes,
        "max_genes": args.max_genes,
        "max_pct_mt": args.max_pct_mt,
        "min_cells_per_gene": args.min_cells_per_gene,
        "n_top_genes": args.n_top_genes,
        "target_sum": args.target_sum,
        "hvg_flavor": args.hvg_flavor,
        "skip_target_gene_retention": args.skip_target_gene_retention,
        "n_hvg_selected": n_hvg_selected,
        "n_target_genes_present": n_target_genes,
        "n_target_overlap": n_target_overlap,
        "n_final_genes_selected": int(selection_mask.sum()),
    }

    adata.write_h5ad(args.output_h5ad)

    summary = {
        "raw_cells": raw_cells,
        "matched_metadata_cells": int(len(adata.obs_names)),
        "metadata_dropped": metadata_dropped,
        "after_good_coverage_filter": after_coverage_filter,
        "after_single_cell_filter": after_single_cell_filter,
        "after_qc_filter": int(cell_mask.sum()),
        "final_cells": int(adata.n_obs),
        "final_genes": int(adata.n_vars),
        "n_hvg_selected": n_hvg_selected,
        "n_target_genes_present": n_target_genes,
        "n_target_overlap": n_target_overlap,
        "n_final_genes_selected": int(selection_mask.sum()),
        "perturbation_counts": adata.obs[args.clean_perturbation_col].value_counts().to_dict(),
        "parameters": adata.uns["preprocessing_params"],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved: {args.output_h5ad}")
    print(f"summary: {args.summary_json}")
    print(f"cells: {adata.n_obs}")
    print(f"genes: {adata.n_vars}")


if __name__ == "__main__":
    main()
