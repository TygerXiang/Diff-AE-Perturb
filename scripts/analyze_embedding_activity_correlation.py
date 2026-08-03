from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.data import CONTROL_PATTERNS, canonical_guide_keys, extract_target_label, is_control_label, load_guide_activity_lookup
from perturbdiff.guide_features import (
    build_notebook_cnn_feature_tensor,
    find_record_for_label,
    guide_cnn_feature_dim,
    load_guide_feature_records,
)
from perturbdiff.model import PerturbDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate sgRNA perturbation embeddings with activity annotations.")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--activity-table", type=Path, required=True)
    parser.add_argument("--activity-col", choices=["relative_activity_day5", "relative_activity_day10"], default="relative_activity_day10")
    parser.add_argument("--guide-activity-cnn-table", type=Path, default=None)
    parser.add_argument("--guide-sequence-length", type=int, default=26)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-strategy", choices=["first_token", "raw"], default="first_token")
    return parser.parse_args()


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> dict[str, float | int | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 3 or np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return {"n": int(xv.size), "stat": None, "pvalue": None}
    if method == "pearson":
        stat, pvalue = pearsonr(xv, yv)
    elif method == "spearman":
        stat, pvalue = spearmanr(xv, yv)
    else:
        raise ValueError(method)
    return {"n": int(xv.size), "stat": float(stat), "pvalue": float(pvalue)}


def pairwise_rank_agreement(rows: list[dict[str, object]], x_key: str, y_key: str) -> dict[str, float | int | None]:
    by_target: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_target.setdefault(str(row["target"]), []).append(row)
    concordant = 0
    discordant = 0
    usable = 0
    groups = 0
    for group in by_target.values():
        vals = [
            (float(row[x_key]), float(row[y_key]))
            for row in group
            if row.get(x_key) is not None and row.get(y_key) is not None
        ]
        vals = [(x, y) for x, y in vals if np.isfinite(x) and np.isfinite(y)]
        if len(vals) < 2:
            continue
        groups += 1
        x = np.asarray([v[0] for v in vals], dtype=np.float64)
        y = np.asarray([v[1] for v in vals], dtype=np.float64)
        for idx in range(len(vals) - 1):
            dx = x[idx + 1 :] - x[idx]
            dy = y[idx + 1 :] - y[idx]
            mask = (dx != 0) & (dy != 0)
            signs = np.sign(dx[mask] * dy[mask])
            concordant += int((signs > 0).sum())
            discordant += int((signs < 0).sum())
            usable += int(mask.sum())
    if usable == 0:
        return {"groups_used": int(groups), "usable_pairs": 0, "agreement_fraction": None, "kendall_like": None}
    return {
        "groups_used": int(groups),
        "usable_pairs": int(usable),
        "concordant_pairs": int(concordant),
        "discordant_pairs": int(discordant),
        "agreement_fraction": float(concordant / usable),
        "kendall_like": float((concordant - discordant) / usable),
    }


def lookup_activity(label: str, lookup: dict[str, float]) -> float | None:
    for key in canonical_guide_keys(label):
        value = lookup.get(key)
        if value is not None:
            return float(value)
    return None


def build_feature(label: str, records_by_key: dict, sequence_length: int) -> torch.Tensor:
    feature_dim = guide_cnn_feature_dim(sequence_length=sequence_length)
    record = find_record_for_label(label, records_by_key)
    if record is None:
        arr = np.zeros(feature_dim, dtype=np.float32)
    else:
        arr = build_notebook_cnn_feature_tensor(record, sequence_length=sequence_length).reshape(-1)
    return torch.from_numpy(arr).unsqueeze(0)


def predicted_activity_for_label(
    model: PerturbDiffusionModel,
    label: str,
    perturbation_id: int,
    records_by_key: dict | None,
    sequence_length: int,
) -> float | None:
    if getattr(model, "guide_strength", None) is None or getattr(model, "guide_doser", None) is None:
        return None
    if records_by_key is None:
        return None
    feature = build_feature(label, records_by_key, sequence_length)
    target_id = torch.tensor([int(perturbation_id)], dtype=torch.long)
    with torch.no_grad():
        strength = model.guide_strength(feature)
        activity = model.guide_doser(strength, target_id)
    return float(activity.squeeze().detach().cpu())


def analyze_checkpoint(
    checkpoint: Path,
    name: str,
    *,
    activity_lookup: dict[str, float],
    records_by_key: dict | None,
    sequence_length: int,
    target_strategy: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mapping = {str(key): int(value) for key, value in state["perturbation_mapping"].items()}
    weight = state["model_state_dict"]["perturbation.embedding.weight"].detach().cpu().float()
    model = None
    if getattr(state["model_config"].guide_encoder, "enabled", False):
        model = PerturbDiffusionModel(state["model_config"]).cpu()
        model.load_state_dict(state["model_state_dict"], strict=False)
        model.eval()
    id_to_label = {idx: label for label, idx in mapping.items()}
    rows: list[dict[str, object]] = []
    for idx in range(1, weight.shape[0]):
        label = id_to_label.get(idx)
        if label is None or label == "*" or is_control_label(label, control_labels=CONTROL_PATTERNS):
            continue
        vec = weight[idx].numpy()
        activity = lookup_activity(label, activity_lookup)
        pred_activity = (
            predicted_activity_for_label(model, label, idx, records_by_key, sequence_length)
            if model is not None
            else None
        )
        rows.append(
            {
                "method": name,
                "id": int(idx),
                "label": label,
                "target": extract_target_label(label, strategy=target_strategy),
                "embedding_norm": float(np.linalg.norm(vec)),
                "activity": activity,
                "predicted_activity": pred_activity,
            }
        )
    norm = np.asarray([float(row["embedding_norm"]) for row in rows], dtype=np.float64)
    activity = np.asarray([np.nan if row["activity"] is None else float(row["activity"]) for row in rows], dtype=np.float64)
    predicted = np.asarray([np.nan if row["predicted_activity"] is None else float(row["predicted_activity"]) for row in rows], dtype=np.float64)
    summary = {
        "checkpoint": str(checkpoint),
        "n_guides": int(len(rows)),
        "embedding_norm_vs_activity": {
            "pearson": safe_corr(norm, activity, "pearson"),
            "spearman": safe_corr(norm, activity, "spearman"),
        },
        "embedding_norm_vs_predicted_activity": {
            "pearson": safe_corr(norm, predicted, "pearson"),
            "spearman": safe_corr(norm, predicted, "spearman"),
        },
        "predicted_activity_vs_activity": {
            "pearson": safe_corr(predicted, activity, "pearson"),
            "spearman": safe_corr(predicted, activity, "spearman"),
        },
        "within_target_rank_agreement": {
            "embedding_norm_vs_activity": pairwise_rank_agreement(rows, "embedding_norm", "activity"),
            "embedding_norm_vs_predicted_activity": pairwise_rank_agreement(rows, "embedding_norm", "predicted_activity"),
            "predicted_activity_vs_activity": pairwise_rank_agreement(rows, "predicted_activity", "activity"),
        },
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    if len(args.checkpoint) != len(args.name):
        raise ValueError("--checkpoint and --name must have the same length")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    activity_lookup = load_guide_activity_lookup(args.activity_table, activity_col=args.activity_col)
    records_by_key = (
        load_guide_feature_records([args.guide_activity_cnn_table])
        if args.guide_activity_cnn_table is not None
        else None
    )
    all_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for checkpoint, name in zip(args.checkpoint, args.name):
        rows, summary = analyze_checkpoint(
            checkpoint,
            name,
            activity_lookup=activity_lookup,
            records_by_key=records_by_key,
            sequence_length=args.guide_sequence_length,
            target_strategy=args.target_strategy,
        )
        all_rows.extend(rows)
        summaries[name] = summary

    rows_path = args.output_dir / "embedding_activity_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "id", "label", "target", "embedding_norm", "activity", "predicted_activity"],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"rows: {rows_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
