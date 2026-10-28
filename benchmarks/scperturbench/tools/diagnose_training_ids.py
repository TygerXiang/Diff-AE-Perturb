#!/usr/bin/env python3
"""Reproduce the training data path exactly and report id ranges and vocabulary sizes.

Answers the questions the crash left open:
  * how many entries does existing_mapping accumulate across train -> val -> test?
  * what are the actual min/max perturbation_ids and target_ids per split?
  * does the concatenated mapping ever exceed the embedding table size?
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/root/Diff-AE-Perturb")
SPLITS = Path("/root/perturbench/.scperturbench/inputs/gse132080_squidiff_repro")
sys.path.insert(0, str(REPO / "src"))

from perturbdiff.data import CONTROL_PATTERNS, load_h5ad_perturbation_dataset  # noqa: E402

TRAIN = SPLITS / "squidiff_train.h5ad"
TEST = SPLITS / "squidiff_test.h5ad"


def describe(name: str, dataset, mapping: dict[str, int]) -> None:
    import torch

    pids = torch.as_tensor(dataset.perturbation_ids)
    tids = torch.as_tensor(dataset.target_ids)
    print(
        f"  {name:<6} n={len(dataset):>6}  perturbation_ids[{int(pids.min())}..{int(pids.max())}]"
        f"  target_ids[{int(tids.min())}..{int(tids.max())}]"
        f"  distinct_pids={int(pids.unique().numel())}"
    )


def main() -> int:
    print("=== loading exactly like scripts/train_with_split.py ===")
    train_ds, mapping = load_h5ad_perturbation_dataset(
        TRAIN,
        perturbation_col="guide_identity",
        activity_label_col="guide_identity",
        target_strategy="raw",
        control_labels=CONTROL_PATTERNS,
    )
    print(f"  after train: len(mapping)={len(mapping)} max_id={max(mapping.values())}")
    describe("train", train_ds, mapping)

    val_ds, _ = load_h5ad_perturbation_dataset(
        TRAIN,
        perturbation_col="guide_identity",
        activity_label_col="guide_identity",
        target_strategy="raw",
        control_labels=CONTROL_PATTERNS,
        existing_mapping=mapping,
    )
    describe("val", val_ds, mapping)

    test_ds, _ = load_h5ad_perturbation_dataset(
        TEST,
        perturbation_col="guide_identity",
        activity_label_col="guide_identity",
        target_strategy="raw",
        control_labels=CONTROL_PATTERNS,
        existing_mapping=mapping,
    )
    print(f"  after test : len(mapping)={len(mapping)} max_id={max(mapping.values())}")
    describe("test", test_ds, mapping)

    embedding_rows = len(mapping) + 1
    print(f"\n  embedding rows the model would build: {embedding_rows}")

    # Per-split-independent target vocabularies
    for split_name, path in (("train", TRAIN), ("test", TEST)):
        ds, tmap = load_h5ad_perturbation_dataset(
            path,
            perturbation_col="guide_identity",
            activity_label_col="guide_identity",
            target_strategy="first_token",
            control_labels=CONTROL_PATTERNS,
        )
        import torch

        tids = torch.as_tensor(ds.target_ids)
        print(f"  fresh first_token mapping on {split_name}: vocab={len(tmap)} max_target_id={int(tids.max())}")

    # The decisive check: does the test split introduce ids the train mapping does not have?
    train_ids = {
        str(v) for v in load_h5ad_perturbation_dataset(
            TRAIN,
            perturbation_col="guide_identity",
            activity_label_col="guide_identity",
            target_strategy="raw",
            control_labels=CONTROL_PATTERNS,
        )[1]
    }
    test_only = sorted({str(g) for g in set(mapping) if str(g) not in train_ids})
    print(f"\n  labels added to mapping by val/test: {len(test_only)} (max id {max(mapping.values())})")
    print(f"  examples: {test_only[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
