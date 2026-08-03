from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.gse283614_biology import fit_response_models, hill_curve, linear_curve, quadratic_curve, read_gene_sets
from perturbdiff.model import PerturbDiffusionModel


CURVE_SPECS = (
    ("ESC_cluster1_score_vs_NANOG", "ESC", "NANOG", "pluripotency_drop", "linear"),
    ("ESC_cluster2_score_vs_OCT4", "ESC", "OCT4", "pluripotency_drop", "hill"),
    ("DE_differentiation_score_vs_NANOG", "DE", "NANOG", "de_gain", "quadratic"),
    ("DE_differentiation_score_vs_OCT4", "DE", "OCT4", "de_gain", "quadratic"),
    ("DE_off-target_score_vs_NANOG", "DE", "NANOG", "ne_gain", "hill"),
    ("DE_off-target_score_vs_OCT4", "DE", "OCT4", "ne_gain", "hill"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate out-of-fold GSE283614 dose predictions.")
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gene-set-workbook", type=Path, required=True)
    parser.add_argument("--observed-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", nargs="+", default=["ESC", "DE"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--max-control-cells", type=int, default=128)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compatible_config_from_state(state: dict) -> object:
    cfg = state["model_config"]
    state_dict = state["model_state_dict"]
    has_activity_condition = any(key.startswith("activity_condition.") for key in state_dict)
    if getattr(cfg.guide_encoder, "enabled", False):
        cfg = replace(
            cfg,
            guide_encoder=replace(
                cfg.guide_encoder,
                explicit_condition=has_activity_condition,
                activity_scales_delta=not has_activity_condition,
            ),
        )
    return cfg


def load_model(path: Path, device: torch.device) -> tuple[PerturbDiffusionModel, dict[str, int]]:
    state = torch.load(path, map_location=device, weights_only=False)
    model = PerturbDiffusionModel(compatible_config_from_state(state)).to(device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    return model, {str(key): int(value) for key, value in state["perturbation_mapping"].items()}


def vector_mean(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).ravel()
    return np.asarray(matrix).mean(axis=0)


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def module_scores(
    mean: np.ndarray,
    control_mean: np.ndarray,
    indices: dict[str, list[int]],
) -> dict[str, float]:
    deltas = {
        name: float(np.mean(mean[idx]) - np.mean(control_mean[idx])) if idx else float("nan")
        for name, idx in indices.items()
    }
    return {
        **{f"{name}_delta_control": value for name, value in deltas.items()},
        "pluripotency_drop": -deltas.get("pluripotency", float("nan")),
        "de_gain": deltas.get("de", float("nan")),
        "ne_gain": deltas.get("ne", float("nan")),
    }


def seed_for(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def predict_mean(
    model: PerturbDiffusionModel,
    mapping: dict[str, int],
    controls: np.ndarray,
    *,
    target: str,
    knockdown: float,
    noise_seed: int,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    target_id = mapping.get(target)
    if target_id is None:
        raise KeyError(f"Target {target!r} missing from checkpoint mapping {sorted(mapping)}")
    x_ctrl = torch.from_numpy(controls.astype(np.float32, copy=False)).to(device)
    ids = torch.full((len(controls),), int(target_id), dtype=torch.long, device=device)
    features = None
    if getattr(model, "guide_doser", None) is not None:
        features = torch.full((len(controls), 1), float(knockdown), dtype=x_ctrl.dtype, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)
    noise = torch.randn(x_ctrl.shape, generator=generator, device=device, dtype=x_ctrl.dtype)
    prediction = model.sample_counterfactual(
        x_ctrl,
        ids,
        guide_features=features,
        steps=steps,
        noise=noise,
    )
    return prediction.detach().cpu().numpy().mean(axis=0)


def curve_fit_rows(table: pd.DataFrame, method: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fit_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for name, state, target, score, expected in CURVE_SPECS:
        frame = table[(table["state"] == state) & (table["dose_target"] == target)]
        y = frame[f"{method}_{score}"].to_numpy(dtype=float)
        span = float(np.nanmax(y) - np.nanmin(y))
        normalized = (y - float(np.nanmin(y))) / span if span > 0 else np.zeros_like(y)
        models = ("linear", "hill") if state == "ESC" else ("linear", "quadratic", "hill")
        fits = fit_response_models(frame["tf_remaining"], normalized, models=models)
        winner = next(row for row in fits if row["selected_aic"])
        quadratic = next((row for row in fits if row["model"] == "quadratic"), None)
        passed = str(winner["model"]) == expected
        if expected == "quadratic":
            passed = passed and bool(quadratic and quadratic.get("inverted_u", False))
        elif expected == "hill":
            passed = (
                passed
                and bool(winner.get("threshold_in_observed_range", False))
                and float(winner.get("hill_coefficient", 0.0)) > 1.0
            )
        fit_rows.extend({"response": name, "method": method, **row} for row in fits)
        summaries.append(
            {
                "response": name,
                "method": method,
                "expected_model": expected,
                "selected_model": winner["model"],
                "pass": bool(passed),
                "vertex": None if quadratic is None else quadratic.get("vertex"),
                "ed50": winner.get("ed50") if winner["model"] == "hill" else None,
                "hill_coefficient": winner.get("hill_coefficient") if winner["model"] == "hill" else None,
                "hill_threshold_in_observed_range": (
                    winner.get("threshold_in_observed_range") if winner["model"] == "hill" else None
                ),
            }
        )
    return fit_rows, summaries


def paired_bootstrap(
    table: pd.DataFrame,
    *,
    value_column: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | list[float]]:
    values = table[value_column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def essentiality_buffering(
    prediction_vectors: dict[tuple[str, str, str, str], list[np.ndarray]],
    gene_names: list[str],
    gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    index = {gene: idx for idx, gene in enumerate(gene_names)}
    rows: list[dict[str, object]] = []
    for method in ("observed", "target_only", "dose"):
        for state in ("ESC", "DE"):
            lineage_category = "pluripotency" if state == "ESC" else "differentiation"
            for target in ("NANOG", "OCT4"):
                vectors = [
                    np.mean(np.stack(values), axis=0)
                    for (key_method, key_state, key_target, _), values in prediction_vectors.items()
                    if key_method == method and key_state == state and key_target == target
                ]
                if len(vectors) < 3:
                    continue
                matrix = np.stack(vectors)

                def relative_ranges(category: str) -> np.ndarray:
                    indices = [index[gene] for gene in gene_sets.get(category, []) if gene in index]
                    subset = matrix[:, indices]
                    scale = np.maximum(np.abs(subset.mean(axis=0)), 1e-6)
                    return np.ptp(subset, axis=0) / scale

                essential = relative_ranges("essentiality")
                lineage = relative_ranges(lineage_category)
                test = stats.mannwhitneyu(essential, lineage, alternative="less")
                buffered = bool(np.median(essential) < np.median(lineage))
                rows.append(
                    {
                        "method": method,
                        "state": state,
                        "dose_target": target,
                        "lineage_category": lineage_category,
                        "essential_median_relative_dynamic_range": float(np.median(essential)),
                        "lineage_median_relative_dynamic_range": float(np.median(lineage)),
                        "essential_buffered": buffered,
                        "mannwhitney_less_pvalue": float(test.pvalue),
                        "essential_buffered_significant": bool(buffered and test.pvalue < 0.05),
                    }
                )
    return pd.DataFrame(rows)


def strong_suppression_checks(table: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target in ("NANOG", "OCT4"):
        frame = table[(table["state"] == "DE") & (table["dose_target"] == target)]
        low_cut = float(frame["knockdown_fraction"].quantile(0.25))
        high_cut = float(frame["knockdown_fraction"].quantile(0.75))
        middle = frame[(frame["knockdown_fraction"] > low_cut) & (frame["knockdown_fraction"] < high_cut)]
        strong = frame[frame["knockdown_fraction"] >= high_cut]
        de_middle = float(middle["dose_de_gain"].mean())
        de_strong = float(strong["dose_de_gain"].mean())
        ne_middle = float(middle["dose_ne_gain"].mean())
        ne_strong = float(strong["dose_ne_gain"].mean())
        rows.append(
            {
                "dose_target": target,
                "strong_knockdown_cutoff": high_cut,
                "n_middle_guides": len(middle),
                "n_strong_guides": len(strong),
                "de_gain_middle": de_middle,
                "de_gain_strong": de_strong,
                "ne_gain_middle": ne_middle,
                "ne_gain_strong": ne_strong,
                "de_declines_and_ne_rises": bool(de_strong < de_middle and ne_strong > ne_middle),
            }
        )
    return rows


def parameter_concordance(
    curve_summaries: list[dict[str, object]],
    observed_summary: dict[str, object],
) -> list[dict[str, object]]:
    observed = {
        str(row["response"]): row
        for row in observed_summary.get("author_response_curves", [])
    }
    predicted = {
        str(row["response"]): row
        for row in curve_summaries
        if row["method"] == "dose"
    }
    rows: list[dict[str, object]] = []
    oct4_name = "ESC_cluster2_score_vs_OCT4"
    oct4_ci = observed.get(oct4_name, {}).get("bootstrap", {}).get("hill_ed50_ci95")
    oct4_value = predicted.get(oct4_name, {}).get("ed50")
    rows.append(
        {
            "parameter": "OCT4_ESC_ED50",
            "predicted": oct4_value,
            "observed_ci95": oct4_ci,
            "within_observed_ci": bool(
                oct4_ci is not None and oct4_value is not None and oct4_ci[0] <= oct4_value <= oct4_ci[1]
            ),
        }
    )
    for target in ("NANOG", "OCT4"):
        name = f"DE_differentiation_score_vs_{target}"
        ci = observed.get(name, {}).get("bootstrap", {}).get("quadratic_vertex_ci95")
        value = predicted.get(name, {}).get("vertex")
        rows.append(
            {
                "parameter": f"{target}_DE_golden_tf_remaining",
                "predicted": value,
                "observed_ci95": ci,
                "within_observed_ci": bool(ci is not None and value is not None and ci[0] <= value <= ci[1]),
            }
        )
    return rows


def plot_curves(table: pd.DataFrame, fit_table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, (name, state, target, score, _) in zip(axes.flat, CURVE_SPECS):
        frame = table[(table["state"] == state) & (table["dose_target"] == target)].sort_values("tf_remaining")
        axis.scatter(frame["tf_remaining"], frame[f"observed_{score}"], label="observed", color="#333333", s=20)
        axis.scatter(frame["tf_remaining"], frame[f"dose_{score}"], label="dose OOF", color="#2463A6", s=20)
        winner = fit_table[
            (fit_table["response"] == name)
            & (fit_table["method"] == "dose")
            & (fit_table["selected_aic"] == True)
        ].iloc[0]
        y = frame[f"dose_{score}"].to_numpy(dtype=float)
        span = float(np.max(y) - np.min(y))
        grid = np.linspace(float(frame["tf_remaining"].min()), float(frame["tf_remaining"].max()), 200)
        if winner["model"] == "linear":
            normalized = linear_curve(grid, winner["intercept"], winner["slope"])
        elif winner["model"] == "quadratic":
            normalized = quadratic_curve(grid, winner["intercept"], winner["linear"], winner["quadratic"])
        else:
            normalized = hill_curve(grid, winner["ed50"], winner["hill_coefficient"])
        axis.plot(grid, normalized * span + float(np.min(y)), color="#2463A6", linewidth=2)
        axis.set_title(name, fontsize=9)
        axis.set_xlabel("TF expression / control")
        axis.set_ylabel(score)
    axes.flat[0].legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("This evaluator requires anndata") from exc
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dose_table = pd.read_csv(args.cv_dir / "gse283614_tf_dosage.tsv", sep="\t")
    dose_lookup = dose_table.set_index("guide_identity").to_dict("index")
    rows: list[dict[str, object]] = []
    prediction_vectors: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
    gene_names: list[str] | None = None
    gene_sets: dict[str, list[str]] | None = None
    set_indices: dict[str, list[int]] | None = None

    for state in [value.upper() for value in args.states]:
        for fold in args.folds:
            path = args.cv_dir / state.lower() / f"fold_{fold}" / f"gse283614_{state.lower()}_test.h5ad"
            adata = ad.read_h5ad(path)
            if gene_names is None:
                gene_names = list(adata.var_names.astype(str))
                index = {gene: idx for idx, gene in enumerate(gene_names)}
                gene_sets = read_gene_sets(args.gene_set_workbook, gene_names)
                set_indices = {name: [index[gene] for gene in members] for name, members in gene_sets.items()}
            assert set_indices is not None and gene_names is not None
            labels = adata.obs["perturbation_label"].astype(str).to_numpy()
            guides = adata.obs["guide_identity"].astype(str).to_numpy()
            control_matrix = dense(adata.X[labels == "control"]).astype(np.float32, copy=False)
            rng = np.random.default_rng(seed_for(state, fold, "controls"))
            if len(control_matrix) > args.max_control_cells:
                control_matrix = control_matrix[rng.choice(len(control_matrix), args.max_control_cells, replace=False)]
            control_mean = control_matrix.mean(axis=0)
            test_guides = sorted(np.unique(guides[labels != "control"]))
            loaded: dict[tuple[int, str], tuple[PerturbDiffusionModel, dict[str, int]]] = {}
            for seed in args.seeds:
                for method in ("target_only", "dose"):
                    checkpoint = args.run_dir / state.lower() / f"fold_{fold}" / f"seed_{seed}" / method / "checkpoint_best.pt"
                    if not checkpoint.exists():
                        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                    loaded[(seed, method)] = load_model(checkpoint, device)

            for guide in test_guides:
                info = dose_lookup.get(guide)
                if info is None:
                    raise KeyError(f"Guide {guide!r} is missing from the TF dosage table")
                mask = guides == guide
                observed_mean = vector_mean(adata.X[mask])
                observed_scores = module_scores(observed_mean, control_mean, set_indices)
                target_gene = "NANOG" if info["dose_target"] == "NANOG" else "POU5F1"
                metric_indices = np.asarray([idx for idx, gene in enumerate(gene_names) if gene != target_gene])
                base: dict[str, object] = {
                    "state": state,
                    "fold": fold,
                    "guide_identity": guide,
                    "dose_target": info["dose_target"],
                    "tf_remaining": float(info["tf_remaining"]),
                    "knockdown_fraction": float(info["knockdown_fraction"]),
                    "n_test_cells": int(mask.sum()),
                    **{f"observed_{key}": value for key, value in observed_scores.items()},
                }
                method_vectors: dict[str, list[np.ndarray]] = {"target_only": [], "dose": []}
                prediction_vectors[("observed", state, str(info["dose_target"]), guide)] = [observed_mean]
                for seed in args.seeds:
                    common_noise_seed = seed_for(state, fold, guide, seed)
                    for method in ("target_only", "dose"):
                        model, mapping = loaded[(seed, method)]
                        predicted_mean = predict_mean(
                            model,
                            mapping,
                            control_matrix,
                            target=str(info["dose_target"]),
                            knockdown=float(info["knockdown_fraction"]),
                            noise_seed=common_noise_seed,
                            steps=args.ddim_steps,
                            device=device,
                        )
                        method_vectors[method].append(predicted_mean)
                for method, vectors in method_vectors.items():
                    predicted_mean = np.mean(np.stack(vectors), axis=0)
                    scores = module_scores(predicted_mean, control_mean, set_indices)
                    base.update({f"{method}_{key}": value for key, value in scores.items()})
                    true_delta = observed_mean[metric_indices] - control_mean[metric_indices]
                    predicted_delta = predicted_mean[metric_indices] - control_mean[metric_indices]
                    base[f"{method}_mse"] = float(np.square(predicted_mean[metric_indices] - observed_mean[metric_indices]).mean())
                    base[f"{method}_pcc_delta"] = correlation(true_delta, predicted_delta)
                    prediction_vectors[(method, state, str(info["dose_target"]), guide)] = vectors
                relevant = ["pluripotency_drop"] if state == "ESC" else ["de_gain", "ne_gain"]
                for method in ("target_only", "dose"):
                    base[f"{method}_score_abs_error"] = float(
                        np.mean([abs(float(base[f"{method}_{score}"]) - float(base[f"observed_{score}"])) for score in relevant])
                    )
                rows.append(base)
            del loaded
            if device.type == "cuda":
                torch.cuda.empty_cache()

    table = pd.DataFrame(rows).sort_values(["state", "dose_target", "tf_remaining", "guide_identity"])
    table["score_error_improvement"] = table["target_only_score_abs_error"] - table["dose_score_abs_error"]
    table["pcc_delta_improvement"] = table["dose_pcc_delta"] - table["target_only_pcc_delta"]
    table.to_csv(args.output_dir / "oof_guide_predictions.csv", index=False)
    all_fit_rows: list[dict[str, object]] = []
    all_curve_summaries: list[dict[str, object]] = []
    for method in ("observed", "target_only", "dose"):
        fits, summaries = curve_fit_rows(table, method)
        all_fit_rows.extend(fits)
        all_curve_summaries.extend(summaries)
    fit_table = pd.DataFrame(all_fit_rows)
    fit_table.to_csv(args.output_dir / "oof_curve_fits.csv", index=False)
    plot_curves(table, fit_table, args.output_dir / "oof_dose_response_six_panel.png")

    score_improvement = paired_bootstrap(
        table,
        value_column="score_error_improvement",
        n_bootstrap=args.bootstrap,
        seed=7,
    )
    pcc_improvement = paired_bootstrap(
        table,
        value_column="pcc_delta_improvement",
        n_bootstrap=args.bootstrap,
        seed=17,
    )
    dose_curve_pass = all(row["pass"] for row in all_curve_summaries if row["method"] == "dose")
    score_pass = score_improvement["ci95"][0] > 0
    pcc_pass = pcc_improvement["ci95"][0] > 0
    observed_summary = json.loads(args.observed_summary.read_text(encoding="utf-8"))
    assert gene_names is not None and gene_sets is not None
    buffering = essentiality_buffering(prediction_vectors, gene_names, gene_sets)
    buffering.to_csv(args.output_dir / "oof_essentiality_buffering.csv", index=False)
    buffering_pass = bool(
        len(buffering[buffering["method"] == "dose"]) == 4
        and buffering.loc[
            buffering["method"] == "dose", "essential_buffered_significant"
        ].all()
    )
    suppression = strong_suppression_checks(table)
    suppression_pass = bool(all(row["de_declines_and_ne_rises"] for row in suppression))
    concordance = parameter_concordance(all_curve_summaries, observed_summary)
    parameter_pass = bool(all(row["within_observed_ci"] for row in concordance))
    summary = {
        "n_oof_guides": len(table),
        "states": sorted(table["state"].unique().tolist()),
        "folds": args.folds,
        "seeds": args.seeds,
        "ddim_steps": args.ddim_steps,
        "curve_results": all_curve_summaries,
        "score_error_improvement": score_improvement,
        "pcc_delta_improvement": pcc_improvement,
        "dose_curve_pass": bool(dose_curve_pass),
        "score_improvement_pass": bool(score_pass),
        "pcc_improvement_pass": bool(pcc_pass),
        "essentiality_buffering": buffering.to_dict("records"),
        "essentiality_buffering_pass": buffering_pass,
        "strong_suppression_checks": suppression,
        "strong_suppression_pass": suppression_pass,
        "parameter_concordance": concordance,
        "parameter_concordance_pass": parameter_pass,
        "observed_stage_gate_pass": bool(observed_summary.get("observed_stage_gate_pass", False)),
        "overall_pass": bool(
            observed_summary.get("observed_stage_gate_pass", False)
            and dose_curve_pass
            and score_pass
            and pcc_pass
            and buffering_pass
            and suppression_pass
            and parameter_pass
        ),
        "note": "Full-transcriptome MSE is auxiliary; overall_pass is controlled by biological curves and paired guide-level improvements.",
    }
    (args.output_dir / "oof_biological_validation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
