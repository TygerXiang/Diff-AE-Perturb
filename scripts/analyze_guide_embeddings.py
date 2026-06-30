from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.data import CONTROL_PATTERNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze learned guideRNA perturbation embeddings by target."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint containing perturbation_mapping and model_state_dict.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for analysis artifacts.",
    )
    parser.add_argument(
        "--target-strategy",
        choices=("first_token", "raw"),
        default="first_token",
        help="How to map guide label to target group.",
    )
    parser.add_argument(
        "--control-label",
        action="append",
        dest="control_labels",
        default=None,
        help="Control label(s). May be provided multiple times.",
    )
    parser.add_argument(
        "--top-k-guides",
        type=int,
        default=10,
        help="How many guides to keep in per-target nearest/farthest tables.",
    )
    return parser.parse_args()


def is_control_label(label: str, control_labels: tuple[str, ...]) -> bool:
    text = label.strip().lower()
    return any(pattern in text for pattern in control_labels)


def parse_target(label: str, strategy: str) -> str:
    if strategy == "raw":
        return label
    return label.split("_", 1)[0].strip()


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return x_norm @ x_norm.T


def pairwise_l2_matrix(x: np.ndarray) -> np.ndarray:
    sq = np.sum(x * x, axis=1, keepdims=True)
    dist2 = np.clip(sq + sq.T - 2.0 * (x @ x.T), 0.0, None)
    return np.sqrt(dist2)


def strict_upper_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([], dtype=np.float64)
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mapping = state.get("perturbation_mapping")
    model_state = state.get("model_state_dict", {})
    emb_key = "perturbation.embedding.weight"
    if mapping is None:
        raise ValueError("Checkpoint missing perturbation_mapping.")
    if emb_key not in model_state:
        raise ValueError(f"Checkpoint missing {emb_key}.")

    control_labels = tuple(x.lower() for x in (args.control_labels or CONTROL_PATTERNS))
    weight = model_state[emb_key].detach().cpu().numpy()

    id_to_label = {idx: label for label, idx in mapping.items()}
    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for idx in range(weight.shape[0]):
        if idx == 0:
            label = "control"
        else:
            label = id_to_label.get(idx, f"<missing-{idx}>")
        if idx == 0 or is_control_label(str(label), control_labels):
            target = "control"
        else:
            target = parse_target(str(label), args.target_strategy)
        vec = weight[idx]
        rows.append(
            {
                "id": idx,
                "label": label,
                "target": target,
                "l2_norm": float(np.linalg.norm(vec)),
            }
        )
        grouped[target].append({"id": idx, "label": label, "vec": vec})

    # Global guide-guide relation table (excluding control row 0).
    guide_entries = [r for r in rows if int(r["id"]) != 0]
    guide_ids = [int(r["id"]) for r in guide_entries]
    guide_labels = [str(r["label"]) for r in guide_entries]
    guide_targets = [str(r["target"]) for r in guide_entries]
    guide_mat = np.stack([weight[i] for i in guide_ids], axis=0)

    cos = cosine_matrix(guide_mat)
    l2 = pairwise_l2_matrix(guide_mat)
    relation_csv = args.output_dir / "guide_pairwise_relations.csv"
    with relation_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["guide_i", "target_i", "guide_j", "target_j", "cosine_similarity", "l2_distance", "same_target"]
        )
        n = len(guide_labels)
        for i in range(n):
            for j in range(i + 1, n):
                writer.writerow(
                    [
                        guide_labels[i],
                        guide_targets[i],
                        guide_labels[j],
                        guide_targets[j],
                        float(cos[i, j]),
                        float(l2[i, j]),
                        int(guide_targets[i] == guide_targets[j]),
                    ]
                )

    # Per-target summary.
    target_summary: list[dict[str, object]] = []
    for target, members in sorted(grouped.items(), key=lambda x: x[0]):
        vecs = np.stack([m["vec"] for m in members], axis=0)
        labels = [str(m["label"]) for m in members]
        norms = np.linalg.norm(vecs, axis=1)
        entry: dict[str, object] = {
            "target": target,
            "n_guides": len(members),
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "guide_labels": labels,
        }
        if len(members) >= 2:
            cos_m = cosine_matrix(vecs)
            l2_m = pairwise_l2_matrix(vecs)
            cos_vals = strict_upper_values(cos_m)
            l2_vals = strict_upper_values(l2_m)
            entry["within_target"] = {
                "cosine_mean": float(np.mean(cos_vals)),
                "cosine_std": float(np.std(cos_vals)),
                "cosine_min": float(np.min(cos_vals)),
                "cosine_max": float(np.max(cos_vals)),
                "l2_mean": float(np.mean(l2_vals)),
                "l2_std": float(np.std(l2_vals)),
                "l2_min": float(np.min(l2_vals)),
                "l2_max": float(np.max(l2_vals)),
            }
            tri_idx = np.triu_indices(len(members), k=1)
            pair_items = []
            for a, b in zip(*tri_idx):
                pair_items.append(
                    {
                        "guide_a": labels[a],
                        "guide_b": labels[b],
                        "cosine_similarity": float(cos_m[a, b]),
                        "l2_distance": float(l2_m[a, b]),
                    }
                )
            pair_items.sort(key=lambda x: x["cosine_similarity"], reverse=True)
            entry["most_similar_pairs"] = pair_items[: args.top_k_guides]
            entry["least_similar_pairs"] = pair_items[-args.top_k_guides :]
        target_summary.append(entry)

    # Global same-target vs cross-target distribution.
    same_target_cos = []
    cross_target_cos = []
    same_target_l2 = []
    cross_target_l2 = []
    n = len(guide_targets)
    for i in range(n):
        for j in range(i + 1, n):
            if guide_targets[i] == guide_targets[j]:
                same_target_cos.append(float(cos[i, j]))
                same_target_l2.append(float(l2[i, j]))
            else:
                cross_target_cos.append(float(cos[i, j]))
                cross_target_l2.append(float(l2[i, j]))

    def dist_stats(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "n_embeddings_total": int(weight.shape[0]),
        "n_guides_excluding_control_row0": int(len(guide_entries)),
        "n_targets": int(len(grouped)),
        "target_strategy": args.target_strategy,
        "control_labels_used": list(control_labels),
        "distribution": {
            "same_target_cosine": dist_stats(same_target_cos),
            "cross_target_cosine": dist_stats(cross_target_cos),
            "same_target_l2": dist_stats(same_target_l2),
            "cross_target_l2": dist_stats(cross_target_l2),
        },
    }

    (args.output_dir / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "target_summary.json").write_text(
        json.dumps(target_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    guide_csv = args.output_dir / "guide_embedding_norms.csv"
    with guide_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "target", "l2_norm"])
        for r in rows:
            writer.writerow([r["id"], r["label"], r["target"], r["l2_norm"]])

    print(f"saved: {args.output_dir / 'embedding_summary.json'}")
    print(f"saved: {args.output_dir / 'target_summary.json'}")
    print(f"saved: {guide_csv}")
    print(f"saved: {relation_csv}")
    print(json.dumps(summary["distribution"], indent=2))


if __name__ == "__main__":
    main()

