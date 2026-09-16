"""Build one report-ready GSE132080 sgRNA results table."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPREHENSIVE = ROOT / "results/comprehensive_sgrna_table.csv"
PRELIM = ROOT / "runs/gse132080_sgrna_200ep/preliminary"
BENCHMARK = ROOT / "runs/benchmarks/activity_branch_200ep_light_per_guide.csv"
OUTPUT = ROOT / "results/gse132080_diffusion_preliminary_results.csv"
REPORT = ROOT / "results/gse132080_diffusion_preliminary_report.md"


def vector_norm(frame: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return np.sqrt(frame[cols].pow(2).sum(axis=1))


def expression_value(matrix: pd.DataFrame, gene: str, guide: str) -> float:
    if gene not in matrix.index or guide not in matrix.columns:
        return np.nan
    return float(matrix.at[gene, guide])


def main() -> None:
    base = pd.read_csv(COMPREHENSIVE)
    base = base.loc[
        base["dataset"].eq("GSE132080")
        & ~base["guide_identity"].isin(["control", "*"])
    ].copy()

    summary = pd.read_csv(PRELIM / "sgrna_summary.csv")
    observed = pd.read_csv(PRELIM / "sgrna_expression_matrix.csv").set_index("gene")
    predicted = pd.read_csv(PRELIM / "sgrna_predicted_expression_matrix.csv").set_index("gene")
    control = pd.read_csv(PRELIM / "expression_control_mean.csv").set_index("gene")["control"]

    zsem = pd.read_csv(PRELIM / "z_sem_conditions.csv")
    zsem = zsem.loc[zsem["z_sem_source"].eq("observed_mean_expression")].copy()
    zsem = zsem.drop(columns=["condition", "target_gene", "n_cells", "z_sem_source"])
    delta = pd.read_csv(PRELIM / "sgrna_delta_vectors.csv").copy()
    delta = delta.rename(columns={col: col.replace("delta_", "sgrna_delta_") for col in delta.columns if col.startswith("delta_")})
    delta = delta.drop(columns=["perturbation_id", "guide_label", "target_gene"])

    summary_keep = [
        "guide_identity", "n_cells", "alignment_target_vs_sgrna",
        "top_down_genes_observed", "top_up_genes_observed",
        "top_down_genes_predicted", "top_up_genes_predicted",
    ]
    result = base.merge(summary[summary_keep], on="guide_identity", how="left", validate="one_to_one")
    result = result.merge(zsem, on="guide_identity", how="left", validate="one_to_one")
    result = result.merge(delta, on="guide_identity", how="left", validate="one_to_one")

    if BENCHMARK.exists():
        benchmark = pd.read_csv(BENCHMARK).rename(columns={"guide": "guide_identity"})
        result = result.merge(benchmark.drop(columns=["target"]), on="guide_identity", how="left", validate="one_to_one")

    result["target_gene_in_expression_panel"] = result["target_gene"].isin(observed.index)
    result["target_gene_control_expression"] = [
        float(control.get(gene, np.nan)) for gene in result["target_gene"]
    ]
    result["target_gene_observed_expression"] = [
        expression_value(observed, gene, guide)
        for gene, guide in zip(result["target_gene"], result["guide_identity"])
    ]
    result["target_gene_observed_delta_vs_control"] = (
        result["target_gene_observed_expression"] - result["target_gene_control_expression"]
    )
    result["target_gene_predicted_expression"] = [
        expression_value(predicted, gene, guide)
        for gene, guide in zip(result["target_gene"], result["guide_identity"])
    ]
    result["target_gene_predicted_delta_vs_control"] = (
        result["target_gene_predicted_expression"] - result["target_gene_control_expression"]
    )
    result["z_sem_l2_norm"] = vector_norm(result, "zsem_")
    result["sgrna_delta_l2_norm"] = vector_norm(result, "sgrna_delta_")
    result["expression_status"] = np.select(
        [
            ~result["target_gene_in_expression_panel"],
            result["n_cells"].isna(),
        ],
        [
            "target_gene_not_in_1019_gene_panel",
            "no_cells_for_guide_in_test_split",
        ],
        default="available",
    )

    readable = [
        "dataset", "guide_identity", "target_gene", "sgrna_sequence",
        "target_genomic_sequence", "pam_sequence", "n_mismatches",
        "mismatch_positions_from_pam", "mismatch_distance_to_pam",
        "min_distance_to_pam", "mismatch_ref_bases", "mismatch_alt_bases",
        "mean_relative_gamma", "virtual_scale", "embedding_norm", "n_cells",
        "target_gene_in_expression_panel", "target_gene_control_expression",
        "target_gene_observed_expression", "target_gene_observed_delta_vs_control",
        "target_gene_predicted_expression", "target_gene_predicted_delta_vs_control",
        "z_sem_l2_norm", "sgrna_delta_l2_norm", "expression_status",
        "alignment_target_vs_sgrna", "top_down_genes_observed", "top_up_genes_observed",
        "top_down_genes_predicted", "top_up_genes_predicted",
    ]
    benchmark_cols = [col for col in result.columns if col.startswith(("no_activity_", "predicted_activity_"))]
    zsem_cols = sorted(col for col in result.columns if col.startswith("zsem_"))
    delta_cols = sorted(col for col in result.columns if re.fullmatch(r"sgrna_delta_\d{3}", col))
    result = result[readable + benchmark_cols + zsem_cols + delta_cols]
    result.to_csv(OUTPUT, index=False)

    manifest = json.loads((PRELIM / "manifest.json").read_text(encoding="utf-8"))
    available = int(result["expression_status"].eq("available").sum())
    no_panel = int(result["expression_status"].eq("target_gene_not_in_1019_gene_panel").sum())
    no_cells = int(result["expression_status"].eq("no_cells_for_guide_in_test_split").sum())
    report = f"""# GSE132080 diffusion model preliminary results

## Scope

- 128 targeting sgRNAs from `comprehensive_sgrna_table.csv`.
- Expression values are normalized values from the 1,019-gene GSE132080 test matrix.
- Target-gene delta is `guide mean expression - control-cell mean expression`.
- Predicted expression is generated from the sgRNA-level checkpoint with 50 DDIM steps.
- `zsem_000..127` is the encoder output for each guide's observed mean expression profile.
- `sgrna_delta_000..127` is the learned sgRNA perturbation embedding from the checkpoint.

## Coverage

- Rows: {len(result)}
- Target-gene expression available: {available}
- Target gene absent from 1,019-gene panel: {no_panel}
- Guide absent from test split: {no_cells}
- Control cells used: {manifest['n_control_cells']}
- Latent dimensions: {manifest['latent_dim']}

## Important interpretation

`target_genomic_sequence` is the 20-nt genomic protospacer, and `pam_sequence` is reported separately. It is not the full genomic span or transcript/CDS of the target gene. `z_sem` is a 128-dimensional cell-expression latent representation, while `sgrna_delta` is the learned 128-dimensional perturbation vector; neither should be interpreted as a single expression measurement.

## Activity input

The primary `gse132080_sgrna_200ep` checkpoint used for this table did not receive activity as a training input (`activity_source` and `conditioning_summary` are absent in its metrics). Its `virtual_scale` is a post-training ratio derived from learned sgRNA embedding norms relative to the fully matched guide.

The separate predicted-activity benchmark uses two 26-nt Table S8 sequences (`genome input` and `sgRNA input`) as a 4 x 26 x 2 one-hot tensor (208 features). A CNN predicts a non-negative guide strength and Table S16 `relative_activity_day10` supplies an auxiliary MSE target with `lambda_strength=0.1`. In the current model implementation the strength is transformed by a target-specific generalized sigmoid, then supplied as an explicit activity embedding to the denoiser when `explicit_condition=True`; optional `activity_scales_delta=True` can additionally multiply the latent perturbation delta.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"wrote {OUTPUT} ({result.shape[0]} rows, {result.shape[1]} columns)")
    print(f"wrote {REPORT}")
    print(result["expression_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
