from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.data import build_virtual_guide_scales_from_embedding, extract_target_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-guide virtual scale s together with embedding norms from a checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--guide-library-table", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mapping = state.get("perturbation_mapping")
    if mapping is None:
        raise ValueError("Checkpoint missing perturbation_mapping")
    emb = state["model_state_dict"]["perturbation.embedding.weight"].detach().cpu()
    scales, rows = build_virtual_guide_scales_from_embedding(
        mapping,
        emb,
        guide_library_tables=args.guide_library_table,
    )

    csv_path = args.output_dir / "guide_virtual_scales.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "label",
                "target",
                "sequence",
                "fully_matched_label",
                "fully_matched_id",
                "s",
                "embedding_l2_norm",
                "fully_matched_l2_norm",
            ]
        )
        writer.writerow([0, "control", "control", "", "control", 0, 0.0, float(emb[0].norm(p=2).item()), 0.0])
        by_target = sorted(rows[1:], key=lambda row: (str(row["target"]), str(row["label"])))
        for row in by_target:
            writer.writerow(
                [
                    int(row["id"]),
                    str(row["label"]),
                    str(row["target"]),
                    str(row.get("sequence", "")),
                    str(row["fully_matched_label"]),
                    int(row["fully_matched_id"]),
                    float(row["s"]),
                    float(row["embedding_l2_norm"]),
                    float(row["fully_matched_l2_norm"]),
                ]
            )

    json_path = args.output_dir / "guide_virtual_scales.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
