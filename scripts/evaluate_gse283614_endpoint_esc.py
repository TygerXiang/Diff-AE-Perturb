from __future__ import annotations

import argparse
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

from perturbdiff.gse283614_biology import fit_response_models, read_gene_sets
from perturbdiff.model import PerturbDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ESC endpoint-to-middle guide predictions.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gene-set-workbook", type=Path, required=True)
    parser.add_argument("--author-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-control-cells", type=int, default=128)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def vector_mean(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel() if sparse.issparse(matrix) else np.asarray(matrix).mean(axis=0)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def load_model(path: Path, device: torch.device) -> tuple[PerturbDiffusionModel, dict[str, int]]:
    state = torch.load(path, map_location=device, weights_only=False)
    model = PerturbDiffusionModel(state["model_config"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, {str(key): int(value) for key, value in state["perturbation_mapping"].items()}


@torch.no_grad()
def predict_mean(
    model: PerturbDiffusionModel,
    mapping: dict[str, int],
    controls: np.ndarray,
    target: str,
    knockdown: float,
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    target_id = mapping[target]
    x = torch.from_numpy(controls.astype(np.float32, copy=False)).to(device)
    ids = torch.full((len(x),), target_id, dtype=torch.long, device=device)
    features = torch.full((len(x), 1), knockdown, dtype=x.dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype)
    result = model.sample_counterfactual(x, ids, guide_features=features, steps=steps, noise=noise)
    return result.detach().cpu().numpy().mean(axis=0)


def correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.std(left[finite]) < 1e-12 or np.std(right[finite]) < 1e-12:
        return float("nan")
    if method == "pearson":
        return float(stats.pearsonr(left[finite], right[finite]).statistic)
    return float(stats.spearmanr(left[finite], right[finite]).statistic)


def bootstrap_mean(values: np.ndarray, n: int, seed: int) -> dict[str, object]:
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def direction_accuracy(observed: np.ndarray, predicted: np.ndarray) -> float:
    keep = (np.abs(observed) > 1e-12) | (np.abs(predicted) > 1e-12)
    return float(np.mean(np.sign(observed[keep]) == np.sign(predicted[keep]))) if keep.any() else float("nan")


def curve_summary(frame: pd.DataFrame, column: str, target: str) -> dict[str, object]:
    subset = frame[frame["dose_target"] == target].sort_values("tf_remaining")
    y = subset[column].to_numpy(float)
    span = float(np.ptp(y))
    normalized = (y - y.min()) / span if span > 0 else np.zeros_like(y)
    if len(subset) < 6:
        return {
            "target": target, "score": column, "n_guides": len(subset),
            "selected_model": None, "fits": [], "reason": "fewer than six blind-test guides",
        }
    fits = fit_response_models(subset["tf_remaining"], normalized, models=("linear", "hill"))
    winner = next(row for row in fits if row["selected_aic"])
    return {
        "target": target,
        "score": column,
        "n_guides": len(subset),
        "selected_model": winner["model"],
        "ed50": winner.get("ed50"),
        "hill_coefficient": winner.get("hill_coefficient"),
        "fits": fits,
    }


def plot_scores(table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {"NANOG": "#087E8B", "OCT4": "#C44536"}
    for axis, (observed, predicted, title) in zip(axes, (
        ("observed_pluripotency_drop", "model_pluripotency_drop", "Recomputed pluripotency score"),
        ("author_esc_score", "model_calibrated_author_score", "Table S4 ESC score"),
    )):
        for target, group in table.groupby("dose_target"):
            axis.scatter(group[observed], group[predicted], label=target, color=colors[target], s=45)
        low = float(min(table[observed].min(), table[predicted].min()))
        high = float(max(table[observed].max(), table[predicted].max()))
        axis.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=1)
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")
        axis.set_title(title)
    axes[0].legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    import anndata as ad

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = ad.read_h5ad(args.split_dir / "esc" / "gse283614_esc_train.h5ad")
    test = ad.read_h5ad(args.split_dir / "esc" / "gse283614_esc_test.h5ad")
    dosage = pd.read_csv(args.split_dir / "gse283614_tf_dosage.tsv", sep="\t")
    author_scores = pd.read_csv(args.author_scores)[["guide_identity", "dose_target", "esc_score"]]
    dosage = dosage.merge(author_scores, on=["guide_identity", "dose_target"], how="left", validate="one_to_one")
    if dosage["esc_score"].isna().any():
        missing = dosage.loc[dosage["esc_score"].isna(), "guide_identity"].tolist()
        raise ValueError(f"Missing author ESC scores for guides: {missing}")
    dosage = dosage.set_index("guide_identity")
    gene_names = list(test.var_names.astype(str))
    if gene_names != list(train.var_names.astype(str)):
        raise ValueError("Train and test gene orders differ")
    gene_sets = read_gene_sets(args.gene_set_workbook, gene_names)
    plur_idx = np.asarray([gene_names.index(gene) for gene in gene_sets["pluripotency"]], dtype=int)

    train_labels = train.obs["perturbation_label"].astype(str).to_numpy()
    test_labels = test.obs["perturbation_label"].astype(str).to_numpy()
    train_guides = train.obs["guide_identity"].astype(str).to_numpy()
    test_guides = test.obs["guide_identity"].astype(str).to_numpy()
    controls = dense(train.X[train_labels == "control"]).astype(np.float32, copy=False)
    rng = np.random.default_rng(args.seed)
    if len(controls) > args.max_control_cells:
        controls = controls[rng.choice(len(controls), args.max_control_cells, replace=False)]
    control_mean = controls.mean(axis=0)

    train_means: dict[str, np.ndarray] = {}
    for guide in np.unique(train_guides[train_labels != "control"]):
        train_means[str(guide)] = vector_mean(train.X[train_guides == guide])

    # Calibrate recomputed module scores to Table S4 using training endpoints only.
    calibration_rows = []
    for guide, mean in train_means.items():
        info = dosage.loc[guide]
        module_drop = float(np.mean(control_mean[plur_idx]) - np.mean(mean[plur_idx]))
        calibration_rows.append((str(info["dose_target"]), module_drop, float(info["esc_score"])))
    calibration: dict[str, tuple[float, float]] = {}
    for target in ("NANOG", "OCT4"):
        pairs = np.asarray([(x, y) for t, x, y in calibration_rows if t == target])
        slope, intercept = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
        calibration[target] = (float(intercept), float(slope))

    model, mapping = load_model(args.checkpoint, torch.device(args.device))
    rows = []
    vector_rows = []
    for guide in sorted(np.unique(test_guides[test_labels != "control"])):
        info = dosage.loc[guide]
        target = str(info["dose_target"])
        dose = float(info["knockdown_fraction"])
        observed = vector_mean(test.X[test_guides == guide])
        predicted = predict_mean(
            model, mapping, controls, target, dose, steps=args.ddim_steps,
            seed=stable_seed(args.seed, guide), device=torch.device(args.device),
        )
        target_guides = [g for g in train_means if str(dosage.loc[g, "dose_target"]) == target]
        target_only = np.mean(np.stack([train_means[g] for g in target_guides]), axis=0)
        ordered = sorted(target_guides, key=lambda g: float(dosage.loc[g, "knockdown_fraction"]))
        nearest_guide = min(ordered, key=lambda g: abs(float(dosage.loc[g, "knockdown_fraction"]) - dose))
        nearest = train_means[nearest_guide]
        low = max((g for g in ordered if float(dosage.loc[g, "knockdown_fraction"]) <= dose), key=lambda g: float(dosage.loc[g, "knockdown_fraction"]), default=ordered[0])
        high = min((g for g in ordered if float(dosage.loc[g, "knockdown_fraction"]) >= dose), key=lambda g: float(dosage.loc[g, "knockdown_fraction"]), default=ordered[-1])
        low_dose, high_dose = float(dosage.loc[low, "knockdown_fraction"]), float(dosage.loc[high, "knockdown_fraction"])
        weight = 0.0 if high_dose == low_dose else (dose - low_dose) / (high_dose - low_dose)
        interpolation = train_means[low] * (1 - weight) + train_means[high] * weight
        base = {
            "guide_identity": guide, "dose_target": target, "n_cells": int(np.sum(test_guides == guide)),
            "tf_remaining": float(info["tf_remaining"]), "knockdown_fraction": dose,
            "author_esc_score": float(info["esc_score"]), "nearest_endpoint_guide": nearest_guide,
            "linear_low_guide": low, "linear_high_guide": high,
        }
        observed_delta = observed - control_mean
        for method, value in {"model": predicted, "target_only": target_only, "nearest_endpoint": nearest, "linear_interpolation": interpolation}.items():
            delta = value - control_mean
            drop = float(np.mean(control_mean[plur_idx]) - np.mean(value[plur_idx]))
            intercept, slope = calibration[target]
            base[f"{method}_pluripotency_drop"] = drop
            base[f"{method}_calibrated_author_score"] = intercept + slope * drop
            base[f"{method}_mse"] = float(np.mean(np.square(value - observed)))
            base[f"{method}_pcc_delta"] = correlation(observed_delta, delta, "pearson")
            base[f"{method}_direction_accuracy_all_genes"] = direction_accuracy(observed_delta, delta)
            base[f"{method}_direction_accuracy_pluripotency"] = direction_accuracy(observed_delta[plur_idx], delta[plur_idx])
            vector_rows.append({"guide_identity": guide, "method": method, "expression": value.astype(np.float32)})
        base["observed_pluripotency_drop"] = float(np.mean(control_mean[plur_idx]) - np.mean(observed[plur_idx]))
        rows.append(base)

    table = pd.DataFrame(rows).sort_values(["dose_target", "knockdown_fraction", "guide_identity"])
    for method in ("model", "target_only", "nearest_endpoint", "linear_interpolation"):
        table[f"{method}_pluripotency_abs_error"] = np.abs(table[f"{method}_pluripotency_drop"] - table["observed_pluripotency_drop"])
        table[f"{method}_author_score_abs_error"] = np.abs(table[f"{method}_calibrated_author_score"] - table["author_esc_score"])
    table.to_csv(args.output_dir / "esc_middle_guide_predictions.csv", index=False)
    plot_scores(table, args.output_dir / "esc_score_observed_vs_predicted.png")
    np.savez_compressed(
        args.output_dir / "esc_middle_predicted_pseudobulk.npz",
        gene_names=np.asarray(gene_names), guide_identity=np.asarray([row["guide_identity"] for row in vector_rows]),
        method=np.asarray([row["method"] for row in vector_rows]), expression=np.stack([row["expression"] for row in vector_rows]),
        control_mean=control_mean.astype(np.float32),
    )

    metrics = {}
    for method in ("model", "target_only", "nearest_endpoint", "linear_interpolation"):
        metrics[method] = {
            "mean_mse": float(table[f"{method}_mse"].mean()),
            "mean_pcc_delta": float(table[f"{method}_pcc_delta"].mean()),
            "mean_direction_accuracy_all_genes": float(table[f"{method}_direction_accuracy_all_genes"].mean()),
            "mean_direction_accuracy_pluripotency": float(table[f"{method}_direction_accuracy_pluripotency"].mean()),
            "pluripotency_mae": float(table[f"{method}_pluripotency_abs_error"].mean()),
            "author_score_mae": float(table[f"{method}_author_score_abs_error"].mean()),
            "pluripotency_pearson": correlation(table["observed_pluripotency_drop"].to_numpy(), table[f"{method}_pluripotency_drop"].to_numpy(), "pearson"),
            "author_score_pearson": correlation(table["author_esc_score"].to_numpy(), table[f"{method}_calibrated_author_score"].to_numpy(), "pearson"),
            "by_target": {
                target: {
                    "n_guides": int(len(group)),
                    "pluripotency_mae": float(group[f"{method}_pluripotency_abs_error"].mean()),
                    "author_score_mae": float(group[f"{method}_author_score_abs_error"].mean()),
                    "pluripotency_pearson": correlation(group["observed_pluripotency_drop"].to_numpy(), group[f"{method}_pluripotency_drop"].to_numpy(), "pearson"),
                    "author_score_pearson": correlation(group["author_esc_score"].to_numpy(), group[f"{method}_calibrated_author_score"].to_numpy(), "pearson"),
                }
                for target, group in table.groupby("dose_target")
            },
        }
    improvements = {}
    for baseline in ("target_only", "nearest_endpoint", "linear_interpolation"):
        for score in ("pluripotency", "author_score"):
            diff = table[f"{baseline}_{score}_abs_error"].to_numpy() - table[f"model_{score}_abs_error"].to_numpy()
            improvements[f"model_vs_{baseline}_{score}_mae"] = bootstrap_mean(diff, args.bootstrap, stable_seed(baseline, score))
    curves = []
    for target in ("NANOG", "OCT4"):
        for column in ("author_esc_score", "observed_pluripotency_drop", "model_pluripotency_drop", "model_calibrated_author_score"):
            curves.append(curve_summary(table, column, target))
    summary = {
        "n_test_guides": len(table), "n_nanog": int((table["dose_target"] == "NANOG").sum()),
        "n_oct4": int((table["dose_target"] == "OCT4").sum()), "n_pluripotency_genes": len(plur_idx),
        "checkpoint": str(args.checkpoint), "ddim_steps": args.ddim_steps,
        "author_score_calibration": {key: {"intercept": value[0], "slope": value[1]} for key, value in calibration.items()},
        "metrics": metrics, "paired_bootstrap_improvements": improvements, "blind_test_curve_fits": curves,
        "caveat": "Author score conversion is an affine calibration fitted only on training endpoint guides; Table S4 test scores were not used for fitting.",
    }
    (args.output_dir / "esc_endpoint_biology_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
