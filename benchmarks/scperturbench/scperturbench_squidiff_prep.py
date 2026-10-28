"""Port of Squidiff's own gene-perturbation preprocessing to GSE132080.

Source recipe
-------------
``Squidiff_reproducibility/prep_pertb_data.ipynb`` (the perturbation notebook) and the
sibling ``prep_*`` notebooks in the same repository.  The perturbation notebook consumes a
matrix that the upstream authors had already normalised (GEARS ``perturb_processed.h5ad``),
so its only data step is:

    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    for gene in <genes of interest>: adata.var.highly_variable[gene] = True
    adata = adata[:, adata.var.highly_variable].copy()
    ... split by ``Group`` (control = 0, one integer per perturbed gene)

The sibling notebooks that start from raw counts
(``prep_drug_screen_data.ipynb``, ``prep_simu_data.ipynb``, ``prep_sciplex_data.ipynb``)
all apply the same normalisation before that: ``sc.pp.normalize_total`` + ``sc.pp.log1p``,
plus a gene filter that drops mitochondrial and ribosomal genes.  This script therefore
reproduces the family recipe end to end:

    raw counts -> QC -> drop MT-/RPS/RPL -> normalize_total(1e4) -> log1p
                -> highly_variable_genes(n_top_genes) -> force genes of interest -> subset

The genes of interest are dataset specific in the paper (erythroid markers for the Norman
combo they study); for a gene-perturbation benchmark the analogous set is the perturbed
genes themselves, which is what ``--force-targets`` selects.  ``--no-force-targets`` gives
the stricter HVG-only panel and is reported alongside for the overlap analysis.

Cell membership is taken from the benchmark's existing unseen-guide split so the Squidiff
run is evaluated on exactly the same cells as the other methods.
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


def read_tsv_rows(path: Path) -> list[list[str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return [line.rstrip("\n").split("\t") for line in handle]


def load_raw_adata(data_dir: Path) -> ad.AnnData:
    matrix_path = next((data_dir / n for n in ("matrix.mtx", "matrix.mtx.gz") if (data_dir / n).exists()), None)
    barcode_path = next((data_dir / n for n in ("barcodes.tsv", "barcodes.tsv.gz") if (data_dir / n).exists()), None)
    gene_path = next(
        (data_dir / n for n in ("genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz") if (data_dir / n).exists()),
        None,
    )
    if matrix_path is None or barcode_path is None or gene_path is None:
        raise FileNotFoundError(f"expected 10x matrix/barcodes/genes files under {data_dir}")
    matrix = mmread(matrix_path).tocsr().transpose().tocsr()
    barcodes = [row[0] for row in read_tsv_rows(barcode_path)]
    gene_rows = read_tsv_rows(gene_path)
    gene_names = [row[1] if len(row) > 1 else row[0] for row in gene_rows]
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=pd.Index(barcodes, name="barcode")),
        var=pd.DataFrame(index=pd.Index(gene_names, name="gene")),
    )
    adata.var_names_make_unique()
    return adata


def clean_perturbation_label(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if "neg_ctrl" in lowered or "non-targeting" in lowered or "non_targeting" in lowered:
        return "control"
    return text.split("_", 1)[0]


def to_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="10x directory of GSE132080")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--train-h5ad", type=Path, required=True, help="benchmark train split (defines cells)")
    parser.add_argument("--test-h5ad", type=Path, required=True, help="benchmark test split (defines cells)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-genes", type=Path, default=None, help="genes.json of the shared panel")
    parser.add_argument("--n-top-genes", type=int, default=1000)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--min-counts", type=float, default=1000)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=10000)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument(
        "--no-force-targets",
        action="store_true",
        help="stop at the plain HVG panel instead of also forcing the perturbed genes",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"recipe": "Squidiff_reproducibility prep_* family, ported to GSE132080"}

    adata = load_raw_adata(args.raw_dir)
    report["raw_cells"] = int(adata.n_obs)
    report["raw_genes"] = int(adata.n_vars)

    metadata = pd.read_csv(args.metadata, index_col=0)
    metadata.index = metadata.index.astype(str)
    keep = [b for b in adata.obs_names if b in set(metadata.index)]
    adata = adata[keep].copy()
    for column in metadata.columns:
        adata.obs[column] = metadata.loc[adata.obs_names, column].to_numpy()
    report["after_metadata_align"] = int(adata.n_obs)

    adata.obs["perturbation_label"] = [clean_perturbation_label(v) for v in adata.obs["guide_identity"]]
    if "good_coverage" in adata.obs:
        adata = adata[to_bool_series(adata.obs["good_coverage"]).to_numpy()].copy()
    if "number_of_cells" in adata.obs:
        adata = adata[
            pd.to_numeric(adata.obs["number_of_cells"], errors="coerce").fillna(np.inf).to_numpy() <= 1
        ].copy()
    report["after_metadata_filters"] = int(adata.n_obs)

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    cell_mask = (
        (adata.obs["total_counts"] >= args.min_counts)
        & (adata.obs["n_genes_by_counts"] >= args.min_genes)
        & (adata.obs["n_genes_by_counts"] <= args.max_genes)
        & (adata.obs["pct_counts_mt"] < args.max_pct_mt)
    )
    adata = adata[cell_mask.to_numpy()].copy()

    x = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    gene_cells = np.asarray((x > 0).sum(axis=0)).reshape(-1)
    non_mt_ribo = np.asarray(~adata.var_names.str.upper().str.startswith(("MT-", "RPS", "RPL")))
    adata = adata[:, ((gene_cells >= args.min_cells_per_gene) & non_mt_ribo)].copy()
    adata.var_names_make_unique()
    report["after_qc"] = [int(adata.n_obs), int(adata.n_vars)]

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)
    report["log1p_scale"] = {
        "mean": float(adata.X[:200].mean()),
        "max": float(adata.X[:200].max()),
        "frac_zero": float((adata.X[:200] == 0).mean()),
    }

    sc.pp.highly_variable_genes(
        adata, n_top_genes=min(args.n_top_genes, adata.n_vars), flavor=args.hvg_flavor, subset=False
    )
    hvg_mask = adata.var["highly_variable"].to_numpy().astype(bool)
    hvg_genes = [str(g) for g, flag in zip(adata.var_names, hvg_mask) if flag]
    report["n_hvg"] = len(hvg_genes)

    target_genes = sorted(
        {str(v) for v in adata.obs["perturbation_label"] if str(v) not in {"control", "*", ""}}
    )
    target_mask = adata.var_names.isin(target_genes)
    report["target_genes_in_matrix"] = int(target_mask.sum())
    if args.no_force_targets:
        selection = hvg_mask
    else:
        selection = hvg_mask | np.asarray(target_mask)
    selected_genes = [str(g) for g, flag in zip(adata.var_names, selection) if flag]
    report["n_selected"] = len(selected_genes)
    report["forced_target_genes_added"] = sorted(set(selected_genes) - set(hvg_genes))
    adata = adata[:, selection].copy()
    report["after_hvg_selection"] = [int(adata.n_obs), int(adata.n_vars)]

    # Cell membership comes from the benchmark split so the comparison stays cell-identical.
    train_split = ad.read_h5ad(args.train_h5ad, backed="r")
    test_split = ad.read_h5ad(args.test_h5ad, backed="r")
    train_barcodes = [str(b) for b in train_split.obs_names]
    test_barcodes = [str(b) for b in test_split.obs_names]
    available = set(map(str, adata.obs_names))
    train_keep = [b for b in train_barcodes if b in available]
    test_keep = [b for b in test_barcodes if b in available]
    report["split_alignment"] = {
        "benchmark_train_cells": len(train_barcodes),
        "squidiff_train_cells": len(train_keep),
        "benchmark_test_cells": len(test_barcodes),
        "squidiff_test_cells": len(test_keep),
    }
    train = adata[train_keep].copy()
    test = adata[test_keep].copy()

    train_targets = sorted(
        {str(v) for v in train.obs["perturbation_label"] if str(v) not in {"control", "*", ""}}
    )
    group_map = {"control": 0}
    for index, gene in enumerate(train_targets, start=1):
        group_map[gene] = index
    for frame in (train, test):
        frame.obs["Group"] = [
            0 if str(v) == "control" else group_map.get(str(v), -1) for v in frame.obs["perturbation_label"]
        ]
    report["group_map_size"] = len(group_map)
    report["test_conditions_without_group"] = sorted(
        {str(v) for v in test.obs["perturbation_label"] if group_map.get(str(v), None) in (None, -1)}
    )

    train.write_h5ad(args.output_dir / "squidiff_train.h5ad")
    test.write_h5ad(args.output_dir / "squidiff_test.h5ad")
    (args.output_dir / "gene_panel.json").write_text(
        json.dumps(
            {
                "recipe": "Squidiff prep_pertb_data: highly_variable_genes + forced target genes",
                "n_top_genes": args.n_top_genes,
                "hvg_flavor": args.hvg_flavor,
                "force_targets": not args.no_force_targets,
                "genes": selected_genes,
                "hvg_only_genes": hvg_genes,
                "target_genes": target_genes,
                "forced_target_genes": sorted(set(selected_genes) - set(hvg_genes)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.benchmark_genes is not None and Path(args.benchmark_genes).exists():
        benchmark = json.loads(Path(args.benchmark_genes).read_text())["genes"]
        shared = sorted(set(selected_genes) & set(benchmark))
        report["benchmark_panel_overlap"] = {
            "benchmark_genes": len(benchmark),
            "squidiff_genes": len(selected_genes),
            "shared": len(shared),
            "only_in_benchmark": sorted(set(benchmark) - set(selected_genes)),
            "only_in_squidiff": sorted(set(selected_genes) - set(benchmark)),
            "hvg_vs_benchmark_shared": len(set(hvg_genes) & set(benchmark)),
        }

    # Numerical equivalence against the benchmark train split: same cells (row order follows the
    # benchmark barcodes) and same genes, so the matrices should agree to float32 precision.
    bench_train = ad.read_h5ad(args.train_h5ad)
    bench_genes = [str(g) for g in bench_train.var_names]
    common = [g for g in selected_genes if g in set(bench_genes)]
    left = train[:, common].X
    right = bench_train[:, common].X
    left = np.asarray(left.todense()) if sparse.issparse(left) else np.asarray(left)
    right = np.asarray(right.todense()) if sparse.issparse(right) else np.asarray(right)
    report["scale_equivalence_vs_benchmark_train"] = {
        "n_genes_compared": len(common),
        "n_cells_compared": int(left.shape[0]),
        "max_abs_diff": float(np.abs(left - right).max()),
        "mean_abs_diff": float(np.abs(left - right).mean()),
        "squidiff_mean": float(left.mean()),
        "benchmark_mean": float(right.mean()),
        "squidiff_max": float(left.max()),
        "benchmark_max": float(right.max()),
    }

    (args.output_dir / "prep_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
