from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse, spatial
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff.data import CONTROL_PATTERNS, extract_target_label, is_control_label
from perturbdiff.guide_features import (
    build_notebook_cnn_feature_tensor,
    find_record_for_label,
    guide_cnn_feature_dim,
    load_guide_feature_records,
)
from perturbdiff.model import PerturbDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight guide-level benchmark for activity conditioning.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--perturbation-col", default="guide_identity")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--guide-activity-cnn-table", type=Path, required=True)
    parser.add_argument("--guide-sequence-length", type=int, default=26)
    parser.add_argument("--min-test-cells-per-guide", type=int, default=10)
    parser.add_argument("--guide-offset", type=int, default=0)
    parser.add_argument("--max-guides", type=int, default=30)
    parser.add_argument("--max-control-cells", type=int, default=256)
    parser.add_argument("--max-metric-cells", type=int, default=256)
    parser.add_argument("--ddim-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="activity_conditioning_light")
    return parser.parse_args()


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def vector_mean(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x).mean(axis=0)


def pearson_delta(true_mean: np.ndarray, pred_mean: np.ndarray, control_mean: np.ndarray) -> float:
    true_delta = true_mean - control_mean
    pred_delta = pred_mean - control_mean
    if float(true_delta.std()) == 0.0 or float(pred_delta.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(true_delta, pred_delta)[0, 1])


def e_distance(x: np.ndarray, y: np.ndarray, *, max_cells: int, rng: np.random.Generator) -> float:
    if x.shape[0] > max_cells:
        x = x[rng.choice(x.shape[0], max_cells, replace=False)]
    if y.shape[0] > max_cells:
        y = y[rng.choice(y.shape[0], max_cells, replace=False)]
    cross = spatial.distance.cdist(x, y, "euclidean").mean()
    within_x = spatial.distance.pdist(x, "euclidean").mean() if x.shape[0] > 1 else 0.0
    within_y = spatial.distance.pdist(y, "euclidean").mean() if y.shape[0] > 1 else 0.0
    return float(2.0 * cross - within_x - within_y)


def load_model(path: Path, device: torch.device) -> tuple[PerturbDiffusionModel, dict[str, int]]:
    state = torch.load(path, map_location=device, weights_only=False)
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
    model = PerturbDiffusionModel(cfg).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, {str(key): int(value) for key, value in state["perturbation_mapping"].items()}


def build_guide_features(
    labels: list[str],
    *,
    model: PerturbDiffusionModel,
    records_by_key: dict,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    if getattr(model, "guide_doser", None) is None:
        return None
    feature_dim = int(model.config.guide_encoder.feature_dim or 0)
    if feature_dim != guide_cnn_feature_dim(sequence_length=sequence_length):
        raise ValueError("Checkpoint guide feature_dim does not match guide_sequence_length")
    rows: list[np.ndarray] = []
    for label in labels:
        record = find_record_for_label(label, records_by_key)
        if record is None:
            rows.append(np.zeros(feature_dim, dtype=np.float32))
        else:
            rows.append(build_notebook_cnn_feature_tensor(record, sequence_length=sequence_length).reshape(-1))
    return torch.from_numpy(np.stack(rows, axis=0)).to(device)


@torch.no_grad()
def predict_guide(
    model: PerturbDiffusionModel,
    mapping: dict[str, int],
    control_cells: torch.Tensor,
    *,
    guide_label: str,
    records_by_key: dict,
    sequence_length: int,
    steps: int,
) -> np.ndarray | None:
    target = extract_target_label(guide_label, strategy="first_token")
    target_id = mapping.get(guide_label)
    if target_id is None:
        target_id = mapping.get(target)
    if target_id is None:
        return None
    target_ids = torch.full((control_cells.shape[0],), target_id, dtype=torch.long, device=control_cells.device)
    guide_features = build_guide_features(
        [guide_label] * control_cells.shape[0],
        model=model,
        records_by_key=records_by_key,
        sequence_length=sequence_length,
        device=control_cells.device,
    )
    pred = model.sample_counterfactual(
        x_ctrl=control_cells,
        target_ids=target_ids,
        guide_features=guide_features,
        steps=steps,
    )
    return pred.detach().cpu().numpy().astype(np.float64, copy=False)


def average(values: list[float], *, higher_is_better: bool = False) -> float:
    vals = [value for value in values if np.isfinite(value)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def jsonable_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, list):
            config[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            config[key] = value
    return config


def main() -> None:
    args = parse_args()
    if len(args.checkpoint) != len(args.name):
        raise ValueError("--checkpoint and --name must be provided the same number of times")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("benchmark_activity_conditioning requires anndata") from exc

    train = ad.read_h5ad(args.train_path)
    test = ad.read_h5ad(args.test_path)
    train_labels = train.obs[args.perturbation_col].astype(str).to_numpy()
    test_labels = test.obs[args.perturbation_col].astype(str).to_numpy()
    train_targets = {
        extract_target_label(label, strategy="first_token")
        for label in train_labels
        if not is_control_label(label, control_labels=CONTROL_PATTERNS)
    }
    control_mask = np.array(
        [is_control_label(label, control_labels=CONTROL_PATTERNS) for label in test_labels],
        dtype=bool,
    )
    control_np = to_dense(test[control_mask].X).astype(np.float32, copy=False)
    if control_np.shape[0] > args.max_control_cells:
        control_np = control_np[rng.choice(control_np.shape[0], args.max_control_cells, replace=False)]
    control_mean = control_np.mean(axis=0).astype(np.float64, copy=False)
    device = torch.device(args.device)
    control_tensor = torch.from_numpy(control_np).to(device)

    value_counts = test.obs[args.perturbation_col].astype(str).value_counts()
    guides: list[str] = []
    for guide, count in value_counts.items():
        guide = str(guide)
        if is_control_label(guide, control_labels=CONTROL_PATTERNS):
            continue
        if int(count) < args.min_test_cells_per_guide:
            continue
        if extract_target_label(guide, strategy="first_token") not in train_targets:
            continue
        guides.append(guide)
    guides = sorted(guides)
    guides = guides[args.guide_offset : args.guide_offset + args.max_guides]

    records_by_key = load_guide_feature_records([args.guide_activity_cnn_table])
    models = {
        name: load_model(checkpoint, device)
        for name, checkpoint in zip(args.name, args.checkpoint)
    }

    per_guide: list[dict[str, object]] = []
    summary_accum: dict[str, dict[str, list[float]]] = {
        name: {"mse": [], "pcc_delta": [], "e_distance": []}
        for name in args.name
    }
    for idx, guide in enumerate(guides, start=1):
        print(f"[{idx}/{len(guides)}] {guide}")
        mask = test.obs[args.perturbation_col].astype(str).to_numpy() == guide
        true_cells = to_dense(test[mask].X).astype(np.float64, copy=False)
        true_mean = true_cells.mean(axis=0)
        row: dict[str, object] = {
            "guide": guide,
            "target": extract_target_label(guide, strategy="first_token"),
            "n_test_cells": int(true_cells.shape[0]),
        }
        for name, (model, mapping) in models.items():
            pred = predict_guide(
                model,
                mapping,
                control_tensor,
                guide_label=guide,
                records_by_key=records_by_key,
                sequence_length=args.guide_sequence_length,
                steps=args.ddim_steps,
            )
            if pred is None:
                continue
            pred_mean = pred.mean(axis=0)
            metrics = {
                "mse": float(np.mean((true_mean - pred_mean) ** 2)),
                "pcc_delta": pearson_delta(true_mean, pred_mean, control_mean),
                "e_distance": e_distance(true_cells, pred, max_cells=args.max_metric_cells, rng=rng),
            }
            for metric_name, value in metrics.items():
                row[f"{name}_{metric_name}"] = value
                summary_accum[name][metric_name].append(value)
        per_guide.append(row)

    summary = []
    for name in args.name:
        summary.append(
            {
                "method": name,
                "n_guides": len(summary_accum[name]["mse"]),
                "mse": average(summary_accum[name]["mse"]),
                "pcc_delta": average(summary_accum[name]["pcc_delta"]),
                "e_distance": average(summary_accum[name]["e_distance"]),
            }
        )

    full_path = args.output_dir / f"{args.output_prefix}_full.json"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.csv"
    per_guide_path = args.output_dir / f"{args.output_prefix}_per_guide.csv"
    full_path.write_text(
        json.dumps(
            {
                "config": jsonable_config(args),
                "summary": summary,
                "per_guide": per_guide,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "n_guides", "mse", "pcc_delta", "e_distance"])
        writer.writeheader()
        writer.writerows(summary)
    fieldnames = sorted({key for row in per_guide for key in row})
    with per_guide_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_guide)
    print(json.dumps(summary, indent=2))
    print(f"saved: {full_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {per_guide_path}")


if __name__ == "__main__":
    main()
