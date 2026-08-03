from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.data import CONTROL_PATTERNS, extract_target_label, is_control_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze direction-only clustering of sgRNA perturbation embeddings."
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-strategy", choices=["first_token", "raw"], default="first_token")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_embeddings(checkpoint: Path, target_strategy: str) -> tuple[list[str], list[str], np.ndarray]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mapping = state.get("perturbation_mapping")
    model_state = state.get("model_state_dict", {})
    emb_key = "perturbation.embedding.weight"
    if mapping is None:
        raise ValueError(f"{checkpoint} is missing perturbation_mapping")
    if emb_key not in model_state:
        raise ValueError(f"{checkpoint} is missing {emb_key}")
    weight = model_state[emb_key].detach().cpu().float().numpy()
    id_to_label = {int(idx): str(label) for label, idx in mapping.items()}
    labels: list[str] = []
    targets: list[str] = []
    vectors: list[np.ndarray] = []
    for idx in range(1, weight.shape[0]):
        label = id_to_label.get(idx)
        if label is None or label == "*" or is_control_label(label, control_labels=CONTROL_PATTERNS):
            continue
        labels.append(label)
        targets.append(extract_target_label(label, strategy=target_strategy))
        vectors.append(weight[idx])
    if not vectors:
        raise ValueError(f"{checkpoint} did not yield any non-control guide embeddings")
    return labels, targets, np.stack(vectors, axis=0)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    unit = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    return unit @ unit.T


def upper_triangle_masks(targets: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(targets)
    tri = np.triu_indices(n, k=1)
    target_arr = np.asarray(targets, dtype=object)
    same = target_arr[tri[0]] == target_arr[tri[1]]
    return tri, same, ~same


def describe(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def compare_groups(same_values: np.ndarray, cross_values: np.ndarray) -> dict[str, object]:
    out = {
        "same_target": describe(same_values),
        "cross_target": describe(cross_values),
        "mean_difference_same_minus_cross": float(same_values.mean() - cross_values.mean()),
        "median_difference_same_minus_cross": float(np.median(same_values) - np.median(cross_values)),
    }
    t_stat, t_p = ttest_ind(same_values, cross_values, equal_var=False)
    u_stat, u_p = mannwhitneyu(same_values, cross_values, alternative="two-sided")
    out.update(
        {
            "welch_t_stat": float(t_stat),
            "welch_t_pvalue": float(t_p),
            "mannwhitney_u": float(u_stat),
            "mannwhitney_pvalue": float(u_p),
        }
    )
    return out


def nearest_neighbor_stats(cos: np.ndarray, targets: list[str], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, object]:
    n = len(targets)
    order = np.argsort(-cos, axis=1)
    target_arr = np.asarray(targets, dtype=object)
    stats: dict[str, object] = {}
    baseline = float(np.mean([(np.sum(target_arr == target) - 1) / (n - 1) for target in target_arr]))
    for k in ks:
        hits = []
        for i in range(n):
            neighbors = [idx for idx in order[i] if idx != i][:k]
            hits.append(float(np.mean(target_arr[neighbors] == target_arr[i])))
        stats[f"top{k}_same_target_fraction"] = {
            "mean": float(np.mean(hits)),
            "baseline_random_mean": baseline,
        }
    return stats


def permutation_pvalues(
    cos: np.ndarray,
    targets: list[str],
    observed_cosine_gap: float,
    observed_top1_hits: int,
    *,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    tri, _, _ = upper_triangle_masks(targets)
    target_arr = np.asarray(targets, dtype=object)
    n = len(targets)
    order = np.argsort(-cos, axis=1)
    cosine_ge = 0
    top1_ge = 0
    for _ in range(permutations):
        perm = rng.permutation(target_arr)
        same = perm[tri[0]] == perm[tri[1]]
        gap = float(cos[tri][same].mean() - cos[tri][~same].mean())
        cosine_ge += int(gap >= observed_cosine_gap)
        hits = 0
        for i in range(n):
            nearest = next(idx for idx in order[i] if idx != i)
            hits += int(perm[nearest] == perm[i])
        top1_ge += int(hits >= observed_top1_hits)
    return {
        "n_permutations": int(permutations),
        "cosine_gap_empirical_p_ge": float((cosine_ge + 1) / (permutations + 1)),
        "top1_same_target_empirical_p_ge": float((top1_ge + 1) / (permutations + 1)),
    }


def per_target_summary(labels: list[str], targets: list[str], cos: np.ndarray) -> list[dict[str, object]]:
    target_arr = np.asarray(targets, dtype=object)
    rows: list[dict[str, object]] = []
    for target in sorted(set(targets)):
        idx = np.flatnonzero(target_arr == target)
        if idx.size < 2:
            continue
        sub = cos[np.ix_(idx, idx)]
        vals = sub[np.triu_indices(idx.size, k=1)]
        cross_idx = np.flatnonzero(target_arr != target)
        cross_vals = cos[np.ix_(idx, cross_idx)].reshape(-1)
        rows.append(
            {
                "target": target,
                "n_guides": int(idx.size),
                "within_cosine_mean": float(vals.mean()),
                "within_cosine_median": float(np.median(vals)),
                "within_cosine_min": float(vals.min()),
                "within_cosine_max": float(vals.max()),
                "cross_cosine_mean": float(cross_vals.mean()),
                "within_minus_cross_mean": float(vals.mean() - cross_vals.mean()),
                "guide_labels": ";".join(labels[i] for i in idx),
            }
        )
    return rows


def analyze_one(
    checkpoint: Path,
    name: str,
    *,
    output_dir: Path,
    target_strategy: str,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    labels, targets, vectors = load_embeddings(checkpoint, target_strategy)
    cos = cosine_matrix(vectors)
    tri, same_mask, cross_mask = upper_triangle_masks(targets)
    pair_values = cos[tri]
    same_values = pair_values[same_mask]
    cross_values = pair_values[cross_mask]
    comparison = compare_groups(same_values, cross_values)
    nn = nearest_neighbor_stats(cos, targets)
    top1_hits = 0
    target_arr = np.asarray(targets, dtype=object)
    order = np.argsort(-cos, axis=1)
    for i in range(len(targets)):
        nearest = next(idx for idx in order[i] if idx != i)
        top1_hits += int(target_arr[nearest] == target_arr[i])
    cosine_gap = float(same_values.mean() - cross_values.mean())
    perm = permutation_pvalues(
        cos,
        targets,
        cosine_gap,
        top1_hits,
        permutations=permutations,
        seed=seed,
    )
    target_rows = per_target_summary(labels, targets, cos)

    method_dir = output_dir / name
    method_dir.mkdir(parents=True, exist_ok=True)
    with (method_dir / "per_target_direction_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "target",
            "n_guides",
            "within_cosine_mean",
            "within_cosine_median",
            "within_cosine_min",
            "within_cosine_max",
            "cross_cosine_mean",
            "within_minus_cross_mean",
            "guide_labels",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(target_rows)

    summary = {
        "checkpoint": str(checkpoint),
        "n_guides": int(len(labels)),
        "n_targets": int(len(set(targets))),
        "target_strategy": target_strategy,
        "cosine_same_vs_cross": comparison,
        "nearest_neighbor": nn,
        "observed_top1_same_target_hits": int(top1_hits),
        "observed_top1_same_target_fraction": float(top1_hits / len(targets)),
        "permutation": perm,
    }
    (method_dir / "direction_clustering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if len(args.checkpoint) != len(args.name):
        raise ValueError("--checkpoint and --name must have the same length")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for idx, (checkpoint, name) in enumerate(zip(args.checkpoint, args.name)):
        summaries[name] = analyze_one(
            checkpoint,
            name,
            output_dir=args.output_dir,
            target_strategy=args.target_strategy,
            permutations=args.permutations,
            seed=args.seed + idx,
        )
    out = {"methods": summaries}
    (args.output_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
