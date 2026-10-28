#!/usr/bin/env python3
"""Measure how much of the held-out test split has a trainable (gradient-receiving) guide embedding.

If a held-out guide label never appears in the training split, its perturbation-embedding row can
never receive a gradient.  This quantifies the consequence for the pooled target-level metrics.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import anndata as ad

SPLITS = Path("/root/perturbench/.scperturbench/inputs/gse132080_squidiff_repro")
TRAIN = SPLITS / "squidiff_train.h5ad"
TEST = SPLITS / "squidiff_test.h5ad"
CONTROL = ("control", "non-targeting", "non_targeted", "neg_ctrl", "ntc")


def is_control(label: str) -> bool:
    lowered = str(label).lower()
    return any(pattern in lowered for pattern in CONTROL)


def main() -> int:
    train = ad.read_h5ad(TRAIN)
    test = ad.read_h5ad(TEST)
    train_guides = set(train.obs["guide_identity"].astype(str))
    train_guides = {g for g in train_guides if not is_control(g)}

    test_guides = test.obs["guide_identity"].astype(str)
    test_labels = test.obs["perturbation_label"].astype(str)

    per_condition: dict[str, dict[str, int]] = defaultdict(lambda: {"covered": 0, "uncovered": 0})
    guide_seen: dict[str, bool] = {}
    for guide, label in zip(test_guides, test_labels):
        if is_control(label) or is_control(guide):
            continue
        seen = guide in train_guides
        guide_seen[guide] = seen
        per_condition[str(label)]["covered" if seen else "uncovered"] += 1

    total = sum(v["covered"] + v["uncovered"] for v in per_condition.values())
    covered = sum(v["covered"] for v in per_condition.values())
    print(f"train guides (targeting, non-control): {len(train_guides)}")
    print(f"test  guides (targeting, non-control): {len(guide_seen)}")
    print(f"test guides also in train            : {sum(guide_seen.values())}")
    print(f"\ntest cells with a trainable guide embedding: {covered}/{total} ({100*covered/total:.1f}%)")
    print("\nper condition:")
    print(f"  {'condition':<10} {'covered':>8} {'uncovered':>10} {'%covered':>9}")
    for condition, counts in sorted(per_condition.items()):
        n = counts["covered"] + counts["uncovered"]
        pct = 100 * counts["covered"] / n if n else 0.0
        flag = "" if counts["uncovered"] == 0 else "   <-- partly untrainable"
        print(f"  {condition:<10} {counts['covered']:>8} {counts['uncovered']:>10} {pct:>8.1f}%{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
