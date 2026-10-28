#!/usr/bin/env python3
"""Report vocabulary coverage of existing GSE132080 checkpoints against the activity tables.

Settles whether the original guide-level runs included the guides that the held-out-guide split
reserves for evaluation, and whether Table S8 / S16 cover them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path("/root/Diff-AE-Perturb")
RUNS = REPO / "runs"
TABLES = REPO / "datasets/GSE132080/41587_2019_387_MOESM3_ESM"
sys.path.insert(0, str(REPO / "src"))
from perturbdiff.data import load_guide_activity_lookup  # noqa: E402
from perturbdiff.guide_features import find_record_for_label, load_guide_feature_records  # noqa: E402

TEST_GUIDES = [
    "ALDOA_ALDOA_+_30077139.23-P1P2_06",
    "GNB2L1_GNB2L1_+_180670873.23-P1P2_00",
    "POLR1D_POLR1D_+_28196016.23-P1_00",
    "TUBB_TUBB_+_30688126.23-P1_00",
    "RPS14_RPS14_+_149829238.23-P1P2_13",
]


def load_mapping(run: str) -> dict[str, int]:
    from json import load

    with (RUNS / run / "perturbation_mapping.json").open(encoding="utf-8") as handle:
        return {str(k): int(v) for k, v in load(handle).items()}


def main() -> int:
    activity = load_guide_activity_lookup(
        TABLES / "Table_S16_perturb-seq_screen_phenotypes.txt",
        activity_col="relative_activity_day10",
    )
    s8 = load_guide_feature_records([TABLES / "Table_S8_machine_learning_input.txt"])

    n16 = len(activity)
    n00 = sum(1 for guide in activity if guide.endswith("_00"))
    print(f"Table S16 labels: {n16} (of which _00 reference guides: {n00})")
    print(f"Table S8 feature records loaded: {len(s8)}\n")

    for run in (
        "gse132080_sgrna_200ep",
        "benchmark_activity_branch_none_200ep",
        "benchmark_activity_branch_predicted_day10_200ep",
        "gse132080_target_200ep",
    ):
        mapping = load_mapping(run)
        # A guide label is usable only if it survives canonical matching into the activity tables.
        matched = sum(
            1 for label in mapping if label in activity or any(
                key in activity for key in (label, "_".join(label.split("_")[1:]))
            )
        )
        held_out_present = [g for g in TEST_GUIDES if g in mapping]
        print(f"=== {run} ===")
        print(f"  mapping entries            : {len(mapping)}")
        print(f"  entries matching Table S16 : {matched}")
        print(f"  held-out test guides present: {len(held_out_present)}/{len(TEST_GUIDES)} {held_out_present}")

    print("\n--- per held-out guide: table coverage ---")
    for guide in TEST_GUIDES:
        print(
            f"  {guide:<52} S16={'yes' if guide in activity else 'NO ':<3}"
            f"  S8={'yes' if find_record_for_label(guide, s8) else 'NO'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
