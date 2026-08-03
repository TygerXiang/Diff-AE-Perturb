from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict", checkpoint)
    if "perturbation.embedding.weight" not in state:
        available = ", ".join(sorted(k for k in state if "embedding" in k)[:20])
        raise KeyError(
            "checkpoint does not contain perturbation.embedding.weight; "
            f"embedding-like keys: {available}"
        )
    return state


def _target_from_label(label: str) -> str:
    return label.split("_", 1)[0]


def export_embeddings(
    *,
    checkpoint_path: Path,
    mapping_path: Path,
    output_prefix: Path,
    expected_guides: int,
) -> dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows = sorted(
        ((int(idx), label) for label, idx in mapping.items() if label != "*" and int(idx) > 0),
        key=lambda item: item[0],
    )
    if len(rows) != expected_guides:
        raise ValueError(f"expected {expected_guides} guide labels, found {len(rows)}")

    checkpoint = _load_checkpoint(checkpoint_path)
    state = _state_dict(checkpoint)
    embedding = state["perturbation.embedding.weight"].detach().cpu().float().numpy()
    max_id = max(idx for idx, _ in rows)
    if embedding.shape[0] <= max_id:
        raise ValueError(
            f"embedding has {embedding.shape[0]} rows but mapping requires row id {max_id}"
        )

    selected = np.stack([embedding[idx] for idx, _ in rows], axis=0)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    npy_path = output_prefix.with_suffix(".npy")
    csv_path = output_prefix.with_suffix(".csv")
    labels_path = output_prefix.with_name(output_prefix.name + "_labels.json")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    np.save(npy_path, selected)
    dim = selected.shape[1]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["perturbation_id", "guide_label", "target", "l2_norm"]
            + [f"emb_{i:03d}" for i in range(dim)]
        )
        for vector, (idx, label) in zip(selected, rows):
            writer.writerow(
                [idx, label, _target_from_label(label), float(np.linalg.norm(vector))]
                + vector.astype(float).tolist()
            )

    labels = [
        {
            "row": row_idx,
            "perturbation_id": idx,
            "guide_label": label,
            "target": _target_from_label(label),
        }
        for row_idx, (idx, label) in enumerate(rows)
    ]
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    summary = {
        "checkpoint": str(checkpoint_path),
        "mapping": str(mapping_path),
        "n_guides": int(selected.shape[0]),
        "embedding_dim": int(selected.shape[1]),
        "excluded_labels": ["*"],
        "npy": str(npy_path),
        "csv": str(csv_path),
        "labels": str(labels_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract guide perturbation embeddings from a PerturbDiff checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-guides", type=int, default=128)
    args = parser.parse_args()

    summary = export_embeddings(
        checkpoint_path=args.checkpoint,
        mapping_path=args.mapping,
        output_prefix=args.output_prefix,
        expected_guides=args.expected_guides,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
