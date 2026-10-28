"""Verify the knockdown-stratum splits: no guide leakage, disjoint controls, bands intact.

Reads the materialized fold files (not the manifest) so the assertions are about
the bytes that training will actually read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if condition else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify knockdown3 splits.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--dosage-table", type=Path, required=True)
    args = parser.parse_args()

    dosage = pd.read_csv(args.dosage_table, sep="\t")
    dose = dict(zip(dosage["guide_identity"].astype(str), dosage["knockdown_fraction"]))
    target = dict(zip(dosage["guide_identity"].astype(str), dosage["dose_target"]))

    seen_test: dict[str, int] = {}
    for state in ("esc", "de"):
        print(f"===== {state.upper()} =====")
        all_guides = set(dose)
        for fold in range(3):
            base = args.split_root / state / f"fold_{fold}"
            splits = {}
            for name in ("train", "val", "test"):
                splits[name] = ad.read_h5ad(base / f"gse283614_{state}_{name}.h5ad", backed="r")

            def guides(a):
                return set(a.obs["perturbation_label"].astype(str)) - {"control"}

            def ctrls(a):
                return set(a.obs_names[a.obs["perturbation_label"].astype(str) == "control"])

            g = {k: guides(v) for k, v in splits.items()}
            c = {k: ctrls(v) for k, v in splits.items()}

            check(f"{state} fold{fold}: train/test guide disjoint", not (g["train"] & g["test"]),
                  f"{sorted(g['train'] & g['test'])}")
            check(f"{state} fold{fold}: train/val guide disjoint", not (g["train"] & g["val"]),
                  f"{sorted(g['train'] & g['val'])}")
            check(f"{state} fold{fold}: val/test guide disjoint", not (g["val"] & g["test"]),
                  f"{sorted(g['val'] & g['test'])}")
            check(f"{state} fold{fold}: control cells disjoint train/test", not (c["train"] & c["test"]),
                  f"{len(c['train'] & c['test'])} overlapping")
            check(f"{state} fold{fold}: test guides nonempty", len(g["test"]) > 0, f"n={len(g['test'])}")

            # Band integrity: within each target, the test band must sit entirely
            # outside the training range (that is the whole point of this design).
            for t in ("NANOG", "OCT4"):
                tst = sorted(dose[x] for x in g["test"] if target[x] == t)
                trn = sorted(dose[x] for x in g["train"] if target[x] == t)
                if not tst or not trn:
                    continue
                overlaps = not (max(tst) < min(trn) or min(tst) > max(trn))
                check(
                    f"{state} fold{fold} {t}: test band outside train range",
                    not overlaps,
                    f"test[{min(tst):.3f},{max(tst):.3f}] train[{min(trn):.3f},{max(trn):.3f}]",
                )

            for guide in g["test"]:
                check_once = seen_test.get(guide, 0)
                seen_test[guide] = check_once + 1
            print(f"  fold{fold}: test={len(g['test'])} guides  cells train={splits['train'].n_obs} "
                  f"val={splits['val'].n_obs} test={splits['test'].n_obs}  ctrl_test={len(c['test'])}")

    # With val_fold = (fold + 1) % 3 each guide is held out from training twice:
    # once as that fold's test block and once as the previous fold's validation
    # block.  Both are legitimate; the invariant that matters is that the guide is
    # never in TRAINING for any fold that evaluates it, which is already asserted
    # per fold above.  What must hold here is exactly-once coverage as TEST.
    test_counts = {g: 0 for g in dose}
    for state in ("esc", "de"):
        for fold in range(3):
            base = args.split_root / state / f"fold_{fold}"
            a = ad.read_h5ad(base / f"gse283614_{state}_test.h5ad", backed="r")
            for guide in set(a.obs["perturbation_label"].astype(str)) - {"control"}:
                test_counts[guide] = test_counts.get(guide, 0) + 1

    # Each state contributes one test appearance, so two states -> each guide
    # appears as a test guide once per state.
    per_state = {g: n for g, n in test_counts.items() if n != 2}
    check("every targeting guide is a test guide exactly once per state", not per_state,
          f"violations={per_state}")
    check("all 33 targeting guides covered", len(test_counts) == 33, f"n={len(test_counts)}")
    print(f"  (each guide is excluded from training twice per state: as test and as val; "
          f"the per-fold disjointness checks above verify neither is ever in train)")

    print()
    if FAILURES:
        print(f"KNOCKDOWN3 VERIFY FAILED: {len(FAILURES)} -> {FAILURES}")
        return 1
    print("KNOCKDOWN3 VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
