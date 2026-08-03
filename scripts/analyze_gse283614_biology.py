from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.gse283614_biology import (
    RESPONSE_SPECS,
    benjamini_hochberg,
    bootstrap_response_fit,
    fit_response_models,
    hill_curve,
    linear_curve,
    linear_effect_summary,
    quadratic_curve,
    read_author_dosage_table,
    read_author_model_selection,
    read_gene_sets,
    validate_dosage_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce GSE283614 dose-response biology before model validation.")
    parser.add_argument("--adata-path", type=Path, required=True)
    parser.add_argument("--dosage-workbook", type=Path, required=True, help="GSE283614 media-4.xlsx")
    parser.add_argument("--guide-library-workbook", type=Path, required=True, help="GSE283614 media-1.xlsx")
    parser.add_argument("--gene-set-workbook", type=Path, required=True, help="GSE283614 media-3.xlsx")
    parser.add_argument("--log2fc-activity-table", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-cells-sensitivity", type=int, default=20)
    parser.add_argument("--skip-gene-fits", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def vector_mean(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).ravel()
    return np.asarray(matrix).mean(axis=0)


def per_cell_module_score(matrix, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.full(matrix.shape[0], np.nan, dtype=float)
    values = matrix[:, indices]
    if sparse.issparse(values):
        return np.asarray(values.mean(axis=1)).ravel()
    return np.asarray(values).mean(axis=1)


def bootstrap_mean_ci(values: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    if n_bootstrap <= 0:
        mean = float(finite.mean())
        return mean, mean
    draws = rng.integers(0, finite.size, size=(n_bootstrap, finite.size))
    means = finite[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def selected_fit(fits: list[dict[str, object]]) -> dict[str, object]:
    return next(row for row in fits if bool(row["selected_aic"]))


def expected_shape_pass(name: str, fits: list[dict[str, object]], expected: str) -> bool:
    winner = selected_fit(fits)
    if str(winner["model"]) != expected:
        return False
    if expected == "quadratic":
        quadratic = next(row for row in fits if row["model"] == "quadratic")
        return bool(quadratic.get("inverted_u", False))
    if expected == "hill":
        return bool(winner.get("threshold_in_observed_range", False)) and float(
            winner.get("hill_coefficient", 0.0)
        ) > 1.0
    return True


def fit_named_curve(
    frame: pd.DataFrame,
    *,
    name: str,
    score_column: str,
    expected: str,
    bootstrap: int,
    seed: int,
    source: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    models = ("linear", "hill") if name.startswith("ESC_") else ("linear", "quadratic", "hill")
    y = frame[score_column].to_numpy(dtype=float)
    if source != "author_score":
        span = float(np.nanmax(y) - np.nanmin(y))
        y = (y - float(np.nanmin(y))) / span if span > 0 else np.zeros_like(y)
    fits = fit_response_models(frame["tf_remaining"], y, models=models)
    rows = [{"response": name, "source": source, **row} for row in fits]
    summary = {
        "response": name,
        "source": source,
        "expected_model": expected,
        "selected_model": str(selected_fit(fits)["model"]),
        "pass": expected_shape_pass(name, fits, expected),
        "bootstrap": bootstrap_response_fit(
            frame["tf_remaining"],
            y,
            n_bootstrap=bootstrap,
            seed=seed,
            models=models,
        ),
    }
    return rows, summary


def build_pseudobulk(
    adata,
    dosage: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    genes = list(adata.var_names.astype(str))
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    set_indices = {name: [gene_index[gene] for gene in members if gene in gene_index] for name, members in gene_sets.items()}
    states = adata.obs["cell_state"].astype(str).to_numpy()
    guides = adata.obs["guide_identity"].astype(str).to_numpy()
    labels = adata.obs["perturbation_label"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    means: dict[tuple[str, str], np.ndarray] = {}
    base_rows: dict[tuple[str, str], dict[str, object]] = {}

    for state in sorted(np.unique(states)):
        state_mask = states == state
        control_mask = state_mask & (labels == "control")
        if not control_mask.any():
            raise ValueError(f"State {state!r} has no controls")
        control_scores = {
            name: float(np.nanmean(per_cell_module_score(adata.X[control_mask], indices)))
            for name, indices in set_indices.items()
        }
        for guide in sorted(np.unique(guides[state_mask])):
            mask = state_mask & (guides == guide)
            matrix = adata.X[mask]
            means[(state, str(guide))] = vector_mean(matrix)
            row: dict[str, object] = {
                "state": state,
                "guide_identity": str(guide),
                "n_cells": int(mask.sum()),
                "low_cell_count": bool(mask.sum() < 20),
            }
            for name, indices in set_indices.items():
                cell_scores = per_cell_module_score(matrix, indices)
                score = float(np.nanmean(cell_scores))
                low, high = bootstrap_mean_ci(cell_scores, n_bootstrap=n_bootstrap, rng=rng)
                row[f"{name}_score"] = score
                row[f"{name}_delta_control"] = score - control_scores[name]
                row[f"{name}_ci95_low"] = low
                row[f"{name}_ci95_high"] = high
                row[f"{name}_n_genes"] = len(indices)
            base_rows[(state, str(guide))] = row

    rows: list[dict[str, object]] = []
    for _, dose_row in dosage.iterrows():
        for state in sorted(np.unique(states)):
            key = (state, str(dose_row["guide_identity"]))
            if key not in base_rows:
                raise KeyError(f"Missing pseudobulk row for {key}")
            row = {**base_rows[key], **dose_row.to_dict()}
            row["pluripotency_drop"] = -float(row.get("pluripotency_delta_control", np.nan))
            row["de_gain"] = float(row.get("de_delta_control", np.nan))
            row["ne_gain"] = float(row.get("ne_delta_control", np.nan))
            rows.append(row)
    return pd.DataFrame(rows), means


def recomputed_specs() -> tuple[tuple[str, str, str, str, str], ...]:
    return (
        ("ESC_cluster1_score_vs_NANOG", "ESC", "NANOG", "pluripotency_drop", "linear"),
        ("ESC_cluster2_score_vs_OCT4", "ESC", "OCT4", "pluripotency_drop", "hill"),
        ("DE_differentiation_score_vs_NANOG", "DE", "NANOG", "de_gain", "quadratic"),
        ("DE_differentiation_score_vs_OCT4", "DE", "OCT4", "de_gain", "quadratic"),
        ("DE_off-target_score_vs_NANOG", "DE", "NANOG", "ne_gain", "hill"),
        ("DE_off-target_score_vs_OCT4", "DE", "OCT4", "ne_gain", "hill"),
    )


def fit_gene_responses(
    pseudobulk: pd.DataFrame,
    means: dict[tuple[str, str], np.ndarray],
    genes: list[str],
    gene_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categories_by_gene: dict[str, list[str]] = {}
    for category, members in gene_sets.items():
        for gene in members:
            categories_by_gene.setdefault(gene, []).append(category)
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    rows: list[dict[str, object]] = []
    for state in ("ESC", "DE"):
        for target in ("NANOG", "OCT4"):
            frame = pseudobulk[(pseudobulk["state"] == state) & (pseudobulk["dose_target"] == target)]
            x = frame["tf_remaining"].to_numpy(dtype=float)
            ordered_keys = [(state, str(guide)) for guide in frame["guide_identity"]]
            for gene in sorted(categories_by_gene):
                idx = gene_index.get(gene)
                if idx is None:
                    continue
                y = np.asarray([means[key][idx] for key in ordered_keys], dtype=float)
                effect = linear_effect_summary(x, y)
                span = float(np.max(y) - np.min(y))
                normalized_y = (y - float(np.min(y))) / span if span > 0 else np.zeros_like(y)
                fits = fit_response_models(x, normalized_y, include_loo=False)
                winner = selected_fit(fits)
                quadratic = next(row for row in fits if row["model"] == "quadratic")
                rows.append(
                    {
                        "state": state,
                        "dose_target": target,
                        "gene": gene,
                        "categories": "|".join(sorted(categories_by_gene[gene])),
                        "selected_model": winner["model"],
                        "inverted_u": bool(quadratic.get("inverted_u", False)),
                        **effect,
                    }
                )
    table = pd.DataFrame(rows)
    table["linear_qvalue"] = np.nan
    for _, indices in table.groupby(["state", "dose_target"]).groups.items():
        table.loc[indices, "linear_qvalue"] = benjamini_hochberg(table.loc[indices, "linear_pvalue"])

    expanded_rows: list[dict[str, object]] = []
    for _, row in table.iterrows():
        for category in str(row["categories"]).split("|"):
            expanded_rows.append({**row.to_dict(), "category": category})
    expanded = pd.DataFrame(expanded_rows)
    summaries: list[dict[str, object]] = []
    for (state, target, category), group in expanded.groupby(["state", "dose_target", "category"]):
        counts = group["selected_model"].value_counts()
        summaries.append(
            {
                "state": state,
                "dose_target": target,
                "category": category,
                "n_genes": len(group),
                "median_dynamic_range": float(group["dynamic_range"].median()),
                "median_relative_dynamic_range": float(group["relative_dynamic_range"].median()),
                "median_abs_standardized_slope": float(group["standardized_slope"].abs().median()),
                "linear_fraction": float(counts.get("linear", 0) / len(group)),
                "quadratic_fraction": float(counts.get("quadratic", 0) / len(group)),
                "hill_fraction": float(counts.get("hill", 0) / len(group)),
                "significant_linear_fraction": float((group["linear_qvalue"] < 0.05).mean()),
            }
        )
    summary = pd.DataFrame(summaries)
    buffered_rows: list[dict[str, object]] = []
    for state in ("ESC", "DE"):
        lineage_category = "pluripotency" if state == "ESC" else "differentiation"
        for target in ("NANOG", "OCT4"):
            essential = expanded[
                (expanded["state"] == state)
                & (expanded["dose_target"] == target)
                & (expanded["category"] == "essentiality")
            ]["relative_dynamic_range"]
            lineage = expanded[
                (expanded["state"] == state)
                & (expanded["dose_target"] == target)
                & (expanded["category"] == lineage_category)
            ]["relative_dynamic_range"]
            test = stats.mannwhitneyu(essential, lineage, alternative="less")
            buffered_rows.append(
                {
                    "state": state,
                    "dose_target": target,
                    "lineage_category": lineage_category,
                    "essential_median_relative_dynamic_range": float(essential.median()),
                    "lineage_median_relative_dynamic_range": float(lineage.median()),
                    "essential_buffered": bool(essential.median() < lineage.median()),
                    "mannwhitney_less_pvalue": float(test.pvalue),
                }
            )
    return table, pd.DataFrame(buffered_rows), summary


def plot_author_curves(dosage: pd.DataFrame, fit_rows: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, spec in zip(axes.flat, RESPONSE_SPECS):
        frame = dosage[dosage["dose_target"] == spec.target].sort_values("tf_remaining")
        axis.scatter(frame["tf_remaining"], frame[spec.score_column], s=22, color="#2463A6", alpha=0.8)
        winner = fit_rows[(fit_rows["response"] == spec.name) & (fit_rows["selected_aic"] == True)].iloc[0]
        grid = np.linspace(float(frame["tf_remaining"].min()), float(frame["tf_remaining"].max()), 200)
        if winner["model"] == "linear":
            values = linear_curve(grid, winner["intercept"], winner["slope"])
        elif winner["model"] == "quadratic":
            values = quadratic_curve(grid, winner["intercept"], winner["linear"], winner["quadratic"])
        else:
            values = hill_curve(grid, winner["ed50"], winner["hill_coefficient"])
        axis.plot(grid, values, color="#B43C2F", linewidth=2)
        axis.set_title(f"{spec.name}\nselected: {winner['model']}", fontsize=9)
        axis.set_xlabel("TF expression / control")
        axis.set_ylabel(spec.score_column)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("This analysis requires anndata") from exc
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.adata_path)
    for column in ("cell_state", "guide_identity", "perturbation_label"):
        if column not in adata.obs:
            raise KeyError(f"Column {column!r} not found in adata.obs")

    dosage = read_author_dosage_table(
        args.dosage_workbook,
        guide_library_workbook=args.guide_library_workbook,
    )
    coverage = validate_dosage_coverage(dosage, adata.obs["guide_identity"].astype(str).unique())
    dose_table = dosage.loc[~dosage["is_control"], [
        "guide_identity",
        "dose_target",
        "tf_expression_pct_control",
        "tf_remaining",
        "knockdown_fraction",
    ]].copy()
    dose_table.to_csv(args.output_dir / "gse283614_tf_dosage.tsv", sep="\t", index=False)
    dosage.to_csv(args.output_dir / "author_guide_scores.csv", index=False)

    author_fit_rows: list[dict[str, object]] = []
    author_summaries: list[dict[str, object]] = []
    for offset, spec in enumerate(RESPONSE_SPECS):
        frame = dosage[dosage["dose_target"] == spec.target]
        rows, summary = fit_named_curve(
            frame,
            name=spec.name,
            score_column=spec.score_column,
            expected=spec.expected_model,
            bootstrap=args.bootstrap,
            seed=args.seed + offset,
            source="author_score",
        )
        author_fit_rows.extend(rows)
        author_summaries.append(summary)
    author_fits = pd.DataFrame(author_fit_rows)
    author_fits.to_csv(args.output_dir / "author_curve_fits.csv", index=False)
    plot_author_curves(dosage, author_fits, args.output_dir / "author_dose_response_six_panel.png")

    gene_sets = read_gene_sets(args.gene_set_workbook, adata.var_names.astype(str))
    pseudobulk, means = build_pseudobulk(
        adata,
        dosage,
        gene_sets,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    pseudobulk.to_csv(args.output_dir / "observed_guide_state_scores.csv", index=False)

    recomputed_fit_rows: list[dict[str, object]] = []
    recomputed_summaries: list[dict[str, object]] = []
    sensitivity_summaries: list[dict[str, object]] = []
    for offset, (name, state, target, score, expected) in enumerate(recomputed_specs()):
        frame = pseudobulk[(pseudobulk["state"] == state) & (pseudobulk["dose_target"] == target)]
        rows, summary = fit_named_curve(
            frame,
            name=name,
            score_column=score,
            expected=expected,
            bootstrap=args.bootstrap,
            seed=args.seed + 100 + offset,
            source="recomputed_gene_set",
        )
        recomputed_fit_rows.extend(rows)
        recomputed_summaries.append(summary)
        sensitivity = frame[frame["n_cells"] >= args.min_cells_sensitivity]
        if len(sensitivity) >= 6:
            _, sensitivity_summary = fit_named_curve(
                sensitivity,
                name=name,
                score_column=score,
                expected=expected,
                bootstrap=args.bootstrap,
                seed=args.seed + 200 + offset,
                source=f"recomputed_n_cells_ge_{args.min_cells_sensitivity}",
            )
            sensitivity_summaries.append(sensitivity_summary)
    pd.DataFrame(recomputed_fit_rows).to_csv(args.output_dir / "recomputed_curve_fits.csv", index=False)

    gene_fit_count = 0
    buffered_pass = None
    if not args.skip_gene_fits:
        gene_fits, buffered_tests, category_summary = fit_gene_responses(
            pseudobulk,
            means,
            list(adata.var_names.astype(str)),
            gene_sets,
        )
        gene_fit_count = len(gene_fits)
        gene_fits.to_csv(args.output_dir / "gene_response_models.csv", index=False)
        buffered_tests.to_csv(args.output_dir / "essentiality_buffering_tests.csv", index=False)
        category_summary.to_csv(args.output_dir / "gene_category_summary.csv", index=False)
        buffered_pass = bool(
            buffered_tests["essential_buffered"].all()
            and (buffered_tests["mannwhitney_less_pvalue"] < 0.05).all()
        )

    log2fc_summary = None
    if args.log2fc_activity_table is not None:
        log2fc = pd.read_csv(args.log2fc_activity_table, sep="\t")
        merged = dose_table.merge(log2fc[["guide_identity", "relative_activity"]], on="guide_identity", how="left")
        mapped = merged.dropna(subset=["relative_activity"])
        correlation = stats.spearmanr(mapped["knockdown_fraction"], mapped["relative_activity"])
        log2fc_summary = {
            "mapped_targeting_guides": len(mapped),
            "missing_targeting_guides": int(merged["relative_activity"].isna().sum()),
            "spearman_rho": float(correlation.statistic),
            "spearman_pvalue": float(correlation.pvalue),
        }
        merged.to_csv(args.output_dir / "tf_dose_vs_log2fc_activity.csv", index=False)

    paper_selection = read_author_model_selection(args.dosage_workbook)
    paper_models = {
        str(row["name"]): str(row["model"]).strip().lower()
        for _, row in paper_selection.iterrows()
        if str(row["name"]) in {spec.name for spec in RESPONSE_SPECS}
    }
    author_gate = bool(all(summary["pass"] for summary in author_summaries))
    table_gate = bool(
        all(paper_models.get(spec.name) == spec.expected_model for spec in RESPONSE_SPECS)
    )
    summary = {
        "adata_path": str(args.adata_path),
        "coverage": coverage,
        "gene_set_sizes": {name: len(genes) for name, genes in gene_sets.items()},
        "author_response_curves": author_summaries,
        "recomputed_response_curves": recomputed_summaries,
        "sensitivity_response_curves": sensitivity_summaries,
        "paper_reported_models": paper_models,
        "author_curve_gate_pass": author_gate,
        "paper_table_gate_pass": table_gate,
        "observed_stage_gate_pass": bool(author_gate and table_gate and buffered_pass is True),
        "recomputed_all_six_pass": bool(all(summary["pass"] for summary in recomputed_summaries)),
        "essentiality_buffering_pass": buffered_pass,
        "gene_response_rows": gene_fit_count,
        "log2fc_sensitivity": log2fc_summary,
        "interpretation": (
            "Proceed to strict model validation only when observed_stage_gate_pass is true. "
            "Recomputed gene-set curves are a preprocessing sensitivity analysis, not a replacement for Table S4."
        ),
    }
    (args.output_dir / "observed_biology_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
