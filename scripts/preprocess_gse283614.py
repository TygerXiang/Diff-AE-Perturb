from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.gse283614 import (
    GuideCallConfig,
    build_log2fc_relative_activity_table,
    build_table_s1_activity_table,
    call_guides_from_matrix,
    guide_family,
    normalize_guide_label,
    read_state_matrices,
    split_feature_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GSE283614 into state-specific AnnData files.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "GSE283614",
        help="Directory containing ESC/ and DE/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "gse283614",
        help="Directory for prepared AnnData files and summaries.",
    )
    parser.add_argument(
        "--gene-set-workbook",
        type=Path,
        default=None,
        help="Optional media-3.xlsx workbook used to retain biologically important genes.",
    )
    parser.add_argument(
        "--table-s1-workbook",
        type=Path,
        default=None,
        help="Optional Table S1 gRNA-library workbook used to export z-score-derived guide activity targets.",
    )
    parser.add_argument(
        "--table-s1-sheet",
        default="0",
        help="Sheet name or zero-based sheet index for --table-s1-workbook.",
    )
    parser.add_argument(
        "--log2fc-activity-workbook",
        type=Path,
        default=None,
        help="Optional reference workbook containing individual-gRNA Log2(Day 9/Day 0) values.",
    )
    parser.add_argument(
        "--log2fc-activity-sheet",
        default="Individual gRNAs",
        help="Sheet name or zero-based sheet index for --log2fc-activity-workbook.",
    )
    parser.add_argument("--min-counts", type=float, default=1000)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=10000)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--hvg-flavor", default="dispersion")
    parser.add_argument("--states", nargs="+", default=("ESC", "DE"))
    parser.add_argument("--min-total-guide-umis", type=int, default=2)
    parser.add_argument("--min-top-guide-umis", type=int, default=2)
    parser.add_argument("--min-top-fraction", type=float, default=0.5)
    parser.add_argument("--min-top-ratio", type=float, default=1.5)
    parser.add_argument("--write-full", action="store_true", default=True)
    parser.add_argument("--no-write-full", action="store_false", dest="write_full")
    return parser.parse_args()


def parse_sheet_name(value: str) -> int | str:
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def load_gene_set_retention(path: Path | None) -> set[str]:
    genes: set[str] = {"POU5F1", "NANOG"}
    if path is None or not path.exists():
        return genes
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError("Reading gene-set workbooks requires openpyxl") from exc

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for ws in workbook.worksheets:
        for row in ws.iter_rows(min_row=4, values_only=True):
            for value in row:
                if value is None:
                    continue
                text = str(value).strip()
                if text and text.upper() not in {"ND", "NA"}:
                    genes.add(text)
    return genes


def as_1d(array_like: object) -> np.ndarray:
    return np.asarray(array_like).reshape(-1)


def calculate_qc_metrics(adata: ad.AnnData) -> None:
    x = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    total_counts = as_1d(x.sum(axis=1)).astype(float)
    n_genes = as_1d((x > 0).sum(axis=1)).astype(int)
    mt_mask = np.asarray(adata.var_names.str.upper().str.startswith("MT-"), dtype=bool)
    mt_counts = as_1d(x[:, mt_mask].sum(axis=1)).astype(float) if mt_mask.any() else np.zeros(x.shape[0])
    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes
    adata.obs["pct_counts_mt"] = np.divide(
        mt_counts * 100.0,
        np.maximum(total_counts, 1.0),
        out=np.zeros_like(total_counts, dtype=float),
        where=total_counts > 0,
    )


def normalize_total_log1p(adata: ad.AnnData, *, target_sum: float) -> None:
    x = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    totals = as_1d(x.sum(axis=1)).astype(float)
    scale = np.divide(
        float(target_sum),
        totals,
        out=np.zeros_like(totals, dtype=float),
        where=totals > 0,
    )
    normalized = x.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    adata.X = normalized


def select_hvg_by_dispersion(adata: ad.AnnData, *, gene_mask: np.ndarray, n_top_genes: int) -> np.ndarray:
    x = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    mean = as_1d(x.mean(axis=0)).astype(float)
    mean_sq = as_1d(x.power(2).mean(axis=0)).astype(float)
    variance = np.maximum(mean_sq - mean**2, 0.0)
    dispersion = np.divide(
        variance,
        np.maximum(mean, 1e-12),
        out=np.zeros_like(variance, dtype=float),
        where=mean > 0,
    )
    dispersion[~gene_mask] = -np.inf
    n_select = min(int(n_top_genes), int(gene_mask.sum()))
    mask = np.zeros(adata.n_vars, dtype=bool)
    if n_select <= 0:
        return mask
    top_idx = np.argsort(dispersion)[-n_select:]
    mask[top_idx] = np.isfinite(dispersion[top_idx])
    return mask


def build_adata_for_state(
    state: str,
    data_dir: Path,
    *,
    guide_call_config: GuideCallConfig,
    min_counts: float,
    min_genes: int,
    max_genes: int,
    max_pct_mt: float,
    min_cells_per_gene: int,
    n_top_genes: int,
    target_sum: float,
    hvg_flavor: str,
    gene_set_retention: set[str],
) -> dict[str, object]:
    matrix, barcodes, feature_ids, feature_names, feature_types = read_state_matrices(data_dir)
    expression, gene_indices, guide_counts, guide_indices = split_feature_matrix(matrix, feature_types)
    gene_names = [str(name).strip() for name in np.asarray(feature_names)[gene_indices].tolist()]
    guide_names = [normalize_guide_label(name) for name in np.asarray(feature_names)[guide_indices].tolist()]

    call_df = call_guides_from_matrix(guide_counts, guide_names, config=guide_call_config)
    guide_status = call_df["guide_assignment_status"].to_numpy()
    assigned_mask = np.isin(guide_status, ["assigned_control", "assigned_guide"])

    obs = pd.DataFrame(index=pd.Index([f"{state}:{barcode}" for barcode in barcodes], name="cell_id"))
    obs["barcode"] = barcodes
    obs["cell_state"] = state
    for col in call_df.columns:
        obs[col] = call_df[col].to_numpy()

    adata_full = ad.AnnData(X=expression.tocsr(), obs=obs, var=pd.DataFrame(index=pd.Index(gene_names, name="gene")))
    adata_full.var["gene_id"] = [str(feature_ids[idx]) for idx in gene_indices]
    adata_full.var["gene_name"] = adata_full.var_names.astype(str)
    adata_full.var_names_make_unique()

    adata_full.var["mt"] = adata_full.var_names.str.upper().str.startswith("MT-")
    calculate_qc_metrics(adata_full)

    qc_mask = (
        (adata_full.obs["total_counts"] >= min_counts)
        & (adata_full.obs["n_genes_by_counts"] >= min_genes)
        & (adata_full.obs["n_genes_by_counts"] <= max_genes)
        & (adata_full.obs["pct_counts_mt"] < max_pct_mt)
    )
    adata_full = adata_full[qc_mask.to_numpy()].copy()
    assigned_mask = assigned_mask[qc_mask.to_numpy()]

    adata_assigned = adata_full[assigned_mask].copy()
    adata_full.layers["counts"] = adata_full.X.copy()
    adata_assigned.layers["counts"] = adata_assigned.X.copy()

    for adata in (adata_full, adata_assigned):
        gene_cells = np.asarray((adata.layers["counts"] > 0).sum(axis=0)).ravel()
        non_mt = ~np.asarray(adata.var_names.str.upper().str.startswith(("MT-", "RPS", "RPL")), dtype=bool)
        gene_mask = (gene_cells >= min_cells_per_gene) & non_mt
        adata.var["passes_gene_qc"] = gene_mask
        normalize_total_log1p(adata, target_sum=target_sum)
        adata.var["highly_variable"] = select_hvg_by_dispersion(
            adata,
            gene_mask=gene_mask,
            n_top_genes=n_top_genes,
        )
        retained = adata.var_names.isin(gene_set_retention)
        adata.var["retain_for_analysis"] = retained
        adata.var["retain"] = (adata.var["highly_variable"].to_numpy().astype(bool) & gene_mask) | retained

    return {
        "state": state,
        "full": adata_full,
        "assigned": adata_assigned,
        "raw_cells": len(barcodes),
        "qc_cells": int(qc_mask.sum()),
        "assigned_cells": int(assigned_mask.sum()),
        "guide_counts": call_df["guide_assignment_status"].value_counts().to_dict(),
        "guide_family_counts": call_df["guide_family"].value_counts().to_dict(),
        "guide_target_counts": call_df["guide_identity"].map(guide_family).value_counts().to_dict(),
        "guide_table": call_df,
        "gene_names_before_subset": int(len(gene_names)),
    }


def write_state_outputs(state_result: dict[str, object], output_dir: Path, *, write_full: bool) -> dict[str, str]:
    state = str(state_result["state"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    assigned_path = output_dir / f"gse283614_{state.lower()}_assigned.h5ad"
    state_result["assigned"].write_h5ad(assigned_path)
    paths["assigned"] = str(assigned_path)
    if write_full:
        full_path = output_dir / f"gse283614_{state.lower()}_full.h5ad"
        state_result["full"].write_h5ad(full_path)
        paths["full"] = str(full_path)
    guide_path = output_dir / f"gse283614_{state.lower()}_guide_assignments.csv"
    state_result["guide_table"].to_csv(guide_path, index=False)
    paths["guide_assignments"] = str(guide_path)
    return paths


def select_common_final_genes(state_results: list[dict[str, object]]) -> list[str]:
    seen: set[str] = set()
    final_genes: list[str] = []
    for result in state_results:
        adata = result["assigned"]
        retained = adata.var["retain"].to_numpy().astype(bool)
        for gene in adata.var_names[retained]:
            if gene not in seen:
                seen.add(str(gene))
                final_genes.append(str(gene))
    return final_genes


def subset_to_final_genes(state_results: list[dict[str, object]], genes: list[str]) -> None:
    for result in state_results:
        for key in ("full", "assigned"):
            adata = result[key]
            present = [gene for gene in genes if gene in adata.var_names]
            result[key] = adata[:, present].copy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gene_set_retention = load_gene_set_retention(args.gene_set_workbook)
    guide_call_config = GuideCallConfig(
        min_total_umis=args.min_total_guide_umis,
        min_top_umis=args.min_top_guide_umis,
        min_top_fraction=args.min_top_fraction,
        min_top_ratio=args.min_top_ratio,
    )

    state_results: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "data_root": str(args.data_root),
        "gene_set_workbook": None if args.gene_set_workbook is None else str(args.gene_set_workbook),
        "table_s1_workbook": None if args.table_s1_workbook is None else str(args.table_s1_workbook),
        "log2fc_activity_workbook": (
            None if args.log2fc_activity_workbook is None else str(args.log2fc_activity_workbook)
        ),
        "parameters": {
            "min_counts": args.min_counts,
            "min_genes": args.min_genes,
            "max_genes": args.max_genes,
            "max_pct_mt": args.max_pct_mt,
            "min_cells_per_gene": args.min_cells_per_gene,
            "n_top_genes": args.n_top_genes,
            "target_sum": args.target_sum,
            "hvg_flavor": args.hvg_flavor,
            "guide_call_config": guide_call_config.__dict__,
        },
        "states": {},
        "retained_gene_count": len(gene_set_retention),
    }

    if args.table_s1_workbook is not None:
        if not args.table_s1_workbook.exists():
            raise FileNotFoundError(f"Missing Table S1 workbook: {args.table_s1_workbook}")
        activity_table = build_table_s1_activity_table(
            args.table_s1_workbook,
            sheet_name=parse_sheet_name(args.table_s1_sheet),
        )
        activity_path = args.output_dir / "gse283614_table_s1_activity.tsv"
        activity_table.to_csv(activity_path, sep="\t", index=False)
        summary["activity_table"] = {
            "path": str(activity_path),
            "recommended_columns": {
                "ESC": "relative_activity_ESC",
                "DE": "relative_activity_DE",
                "combined": "relative_activity_mean",
            },
            "z_score_conversion": "screen_depletion_effect_STATE = -screen_z_score_STATE; relative_activity_signed_STATE = screen_depletion_effect_STATE / within_region_reference_effect; relative_activity_STATE = max(0, relative_activity_signed_STATE). The default reference is the strongest depletion effect in the same Type/Region group.",
            "n_rows": int(activity_table.shape[0]),
        }

    if args.log2fc_activity_workbook is not None:
        if not args.log2fc_activity_workbook.exists():
            raise FileNotFoundError(f"Missing log2FC activity workbook: {args.log2fc_activity_workbook}")
        log2fc_activity = build_log2fc_relative_activity_table(
            args.log2fc_activity_workbook,
            guide_library_path=args.table_s1_workbook,
            sheet_name=parse_sheet_name(args.log2fc_activity_sheet),
        )
        log2fc_activity_path = args.output_dir / "gse283614_log2fc_relative_activity.tsv"
        log2fc_activity.to_csv(log2fc_activity_path, sep="\t", index=False)
        summary["log2fc_activity_table"] = {
            "path": str(log2fc_activity_path),
            "recommended_column": "relative_activity",
            "conversion": "screen_depletion_effect = -Log2(Day 9/Day 0); relative_activity_signed = screen_depletion_effect / within_region_reference_effect; relative_activity = max(0, relative_activity_signed). The default reference is the strongest depletion guide in the same reference region.",
            "mapped_to_gse283614_guides": args.table_s1_workbook is not None,
            "n_rows": int(log2fc_activity.shape[0]),
        }

    for state in args.states:
        state_dir = args.data_root / state
        if not state_dir.exists():
            raise FileNotFoundError(f"Missing state directory: {state_dir}")
        result = build_adata_for_state(
            state,
            state_dir,
            guide_call_config=guide_call_config,
            min_counts=args.min_counts,
            min_genes=args.min_genes,
            max_genes=args.max_genes,
            max_pct_mt=args.max_pct_mt,
            min_cells_per_gene=args.min_cells_per_gene,
            n_top_genes=args.n_top_genes,
            target_sum=args.target_sum,
            hvg_flavor=args.hvg_flavor,
            gene_set_retention=gene_set_retention,
        )
        state_results.append(result)

    final_genes = select_common_final_genes(state_results)
    subset_to_final_genes(state_results, final_genes)

    for result in state_results:
        state = str(result["state"])
        paths = write_state_outputs(result, args.output_dir, write_full=args.write_full)
        summary["states"][state] = {
            "raw_cells": result["raw_cells"],
            "qc_cells": result["qc_cells"],
            "assigned_cells": result["assigned_cells"],
            "guide_assignment_counts": result["guide_counts"],
            "guide_family_counts": result["guide_family_counts"],
            "guide_target_counts": result["guide_target_counts"],
            "paths": paths,
            "n_genes_assigned": int(result["assigned"].n_vars),
            "n_genes_full": int(result["full"].n_vars),
        }

    assigned_adatas = [result["assigned"] for result in state_results]
    full_adatas = [result["full"] for result in state_results]
    combined_assigned = ad.concat(assigned_adatas, join="inner")
    combined_full = ad.concat(full_adatas, join="inner")
    combined_assigned_path = args.output_dir / "gse283614_combined_assigned.h5ad"
    combined_full_path = args.output_dir / "gse283614_combined_full.h5ad"
    combined_assigned.write_h5ad(combined_assigned_path)
    combined_full.write_h5ad(combined_full_path)
    summary["combined"] = {
        "assigned_path": str(combined_assigned_path),
        "full_path": str(combined_full_path),
        "n_cells_assigned": int(combined_assigned.n_obs),
        "n_cells_full": int(combined_full.n_obs),
        "n_genes": int(combined_assigned.n_vars),
        "states": [result["state"] for result in state_results],
    }

    summary_path = args.output_dir / "gse283614_preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {summary_path}")
    print(json.dumps(summary["combined"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
