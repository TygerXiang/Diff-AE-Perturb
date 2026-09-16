"""Build the report-ready GSE132080 predicted-activity branch table."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRELIM = ROOT / "runs/benchmark_activity_branch_predicted_day10_200ep/preliminary"
OUTPUT = ROOT / "results/gse132080_activity_branch_results.csv"


def matrix_value(matrix: pd.DataFrame, gene: str, guide: str) -> float:
    if gene not in matrix.index or guide not in matrix.columns:
        return np.nan
    return float(matrix.at[gene, guide])


def main() -> None:
    base = pd.read_csv(ROOT / "results/comprehensive_sgrna_table.csv")
    base = base.loc[
        base["dataset"].eq("GSE132080")
        & ~base["guide_identity"].isin(["control", "*"])
    ].copy()
    summary = pd.read_csv(PRELIM / "sgrna_summary.csv")
    observed = pd.read_csv(PRELIM / "sgrna_expression_matrix.csv").set_index("gene")
    predicted = pd.read_csv(PRELIM / "sgrna_predicted_expression_matrix.csv").set_index("gene")
    control = pd.read_csv(PRELIM / "expression_control_mean.csv").set_index("gene")["control"]

    zsem = pd.read_csv(PRELIM / "z_sem_conditions.csv")
    zsem = zsem.loc[zsem["z_sem_source"].eq("observed_mean_expression")]
    zsem = zsem.drop(columns=["condition", "target_gene", "n_cells", "z_sem_source"])
    delta = pd.read_csv(PRELIM / "sgrna_delta_vectors.csv")
    delta = delta.rename(columns={c: c.replace("delta_", "sgrna_delta_") for c in delta if c.startswith("delta_")})
    delta = delta.drop(columns=["perturbation_id", "guide_label", "target_gene"])

    keep = [
        "guide_identity", "n_cells", "alignment_target_vs_sgrna",
        "activity_raw_strength", "activity_scaled_strength",
    ]
    result = base.merge(summary[keep], on="guide_identity", how="left", validate="one_to_one")
    result = result.merge(zsem, on="guide_identity", how="left", validate="one_to_one")
    result = result.merge(delta, on="guide_identity", how="left", validate="one_to_one")

    benchmark_path = ROOT / "runs/benchmarks/activity_branch_200ep_light_per_guide.csv"
    benchmark = pd.read_csv(benchmark_path).rename(columns={"guide": "guide_identity"})
    result = result.merge(benchmark.drop(columns=["target"]), on="guide_identity", how="left", validate="one_to_one")

    result["target_gene_in_expression_panel"] = result["target_gene"].isin(observed.index)
    result["target_gene_control_expression"] = [float(control.get(g, np.nan)) for g in result["target_gene"]]
    result["target_gene_observed_expression"] = [
        matrix_value(observed, gene, guide)
        for gene, guide in zip(result["target_gene"], result["guide_identity"])
    ]
    result["target_gene_observed_delta_vs_control"] = (
        result["target_gene_observed_expression"] - result["target_gene_control_expression"]
    )
    result["target_gene_predicted_expression"] = [
        matrix_value(predicted, gene, guide)
        for gene, guide in zip(result["target_gene"], result["guide_identity"])
    ]
    result["target_gene_predicted_delta_vs_control"] = (
        result["target_gene_predicted_expression"] - result["target_gene_control_expression"]
    )
    zcols = sorted(c for c in result if re.fullmatch(r"zsem_\d{3}", c))
    dcols = sorted(c for c in result if re.fullmatch(r"sgrna_delta_\d{3}", c))
    result["z_sem_l2_norm"] = np.sqrt(result[zcols].pow(2).sum(axis=1, min_count=1))
    result["sgrna_delta_l2_norm"] = np.sqrt(result[dcols].pow(2).sum(axis=1, min_count=1))
    result["expression_status"] = np.select(
        [~result["target_gene_in_expression_panel"], result["n_cells"].isna()],
        ["target_gene_not_in_1019_gene_panel", "no_cells_for_guide_in_test_split"],
        default="available",
    )

    readable = [
        "dataset", "guide_identity", "target_gene", "sgrna_sequence",
        "target_genomic_sequence", "pam_sequence", "n_mismatches",
        "mismatch_positions_from_pam", "mismatch_distance_to_pam", "min_distance_to_pam",
        "mismatch_ref_bases", "mismatch_alt_bases", "mean_relative_gamma", "n_cells",
        "activity_raw_strength", "activity_scaled_strength",
        "target_gene_in_expression_panel", "target_gene_control_expression",
        "target_gene_observed_expression", "target_gene_observed_delta_vs_control",
        "target_gene_predicted_expression", "target_gene_predicted_delta_vs_control",
        "z_sem_l2_norm", "sgrna_delta_l2_norm", "expression_status",
        "alignment_target_vs_sgrna",
    ]
    metrics = [c for c in result if c.startswith(("no_activity_", "predicted_activity_"))]
    result = result[readable + metrics + zcols + dcols]
    result.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT} ({result.shape[0]} rows, {result.shape[1]} columns)")
    print(result["expression_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
