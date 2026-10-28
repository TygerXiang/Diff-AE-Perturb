#!/usr/bin/env python3
"""Verify that a target-level perturbation embedding is fully trainable on the held-out-guide split.

With ``--perturbation-col perturbation_label`` the embedding is indexed by target gene.  That is
trainable as long as every evaluated condition's target gene appears in the training split, even
though the *guides* are held out.  Guide-level Table S8 features still vary between guides of the
same target, which is what the predicted-efficiency conditioning consumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad

SPLITS = Path("/root/perturbench/.scperturbench/inputs/gse132080_squidiff_repro")
BENCH = Path("/root/perturbench/.scperturbench/inputs/gse132080_bench/conditions.json")
CONTROL = ("control", "non-targeting", "non_targeted", "neg_ctrl", "ntc", "non-targeting")


def is_control(label: str) -> bool:
    lowered = str(label).lower()
    return any(pattern in lowered for pattern in CONTROL)


def main() -> int:
    import json

    train = ad.read_h5ad(SPLITS / "squidiff_train.h5ad")
    test = ad.read_h5ad(SPLITS / "squidiff_test.h5ad")
    conditions = json.loads(BENCH.read_text(encoding="utf-8"))
    evaluated = [str(c) for c in conditions["evaluated_conditions"]]

    train_labels = train.obs["perturbation_label"].astype(str)
    test_labels = test.obs["perturbation_label"].astype(str)
    train_targets = {label for label in train_labels if not is_control(label)}
    test_targets = {label for label in test_labels if not is_control(label)}

    print(f"train target labels : {len(train_targets)} -> {sorted(train_targets)}")
    print(f"test  target labels : {len(test_targets)} -> {sorted(test_targets)}")
    missing = sorted(set(evaluated) - train_targets)
    print(f"\nevaluated conditions absent from train targets: {missing or 'NONE'}")

    # guide coverage per target on both sides: this is where sgRNA-efficiency variation comes from
    print("\nper evaluated condition (train-side guide diversity is what trains the activity branch):")
    print(f"  {'condition':<10} {'train cells':>11} {'train guides':>12} {'test cells':>10} {'test guides':>11}")
    ok = True
    for condition in evaluated:
        tr = train[train_labels == condition]
        te = test[test_labels == condition]
        n_train_guides = tr.obs["guide_identity"].astype(str).nunique()
        n_test_guides = te.obs["guide_identity"].astype(str).nunique()
        flag = ""
        if tr.n_obs == 0:
            flag = "   <-- NO TRAIN CELLS"
            ok = False
        elif n_train_guides < 2:
            flag = "   <-- single guide: no within-target efficiency contrast"
        print(
            f"  {condition:<10} {tr.n_obs:>11} {n_train_guides:>12} {te.n_obs:>10} {n_test_guides:>11}{flag}"
        )

    n_vocab = len({label for label in train_labels} | {label for label in test_labels})
    print(f"\ntarget vocabulary (train | test labels, incl control): {n_vocab}")
    print("VERDICT:", "trainable target-level embedding covers all evaluated conditions"
          if ok and not missing else "PROBLEM: see flags above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
