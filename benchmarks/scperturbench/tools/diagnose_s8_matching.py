"""Diagnose why some h5ad guide identities do not resolve against Table S8.

Prints, for every guide the activity arm must condition on: the h5ad label, the S8 rows that
share its gene and sgRNA sequence, and which canonical keys exist in the loaded record table.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import anndata as ad

REPO = Path("/root/Diff-AE-Perturb")
WORK = Path("/root/perturbench")
TABLES = REPO / "datasets/GSE132080/41587_2019_387_MOESM3_ESM"
S8 = TABLES / "Table_S8_machine_learning_input.txt"
S16 = TABLES / "Table_S16_perturb-seq_screen_phenotypes.txt"
TRAIN = WORK / ".scperturbench/inputs/gse132080_squidiff_repro/squidiff_train.h5ad"
TEST = WORK / ".scperturbench/inputs/gse132080_squidiff_repro/squidiff_test.h5ad"

sys.path.insert(0, str(REPO / "src"))
from perturbdiff.guide_features import canonical_guide_keys, find_record_for_label, load_guide_feature_records  # noqa: E402


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    records = load_guide_feature_records([S8])
    s8_rows = read_table(S8)
    s16_rows = read_table(S16)

    by_gene_seq: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in s8_rows:
        label = str(list(row.values())[0]).strip()
        by_gene_seq[(row.get("gene", "").strip(), row.get("sgRNA sequence", "").strip().upper())].append(label)

    s16_by_label: dict[str, dict[str, str]] = {}
    for row in s16_rows:
        s16_by_label[str(list(row.values())[0]).strip()] = row

    train = ad.read_h5ad(TRAIN)
    test = ad.read_h5ad(TEST)
    guides = sorted(
        {
            str(g)
            for frame in (train, test)
            for g in frame.obs["guide_identity"].astype(str)
            if g != "*" and "control" not in g.lower() and "non-targeting" not in g.lower()
        }
    )
    print(f"training/targeting guides in h5ad: {len(guides)}")
    print(f"S8 rows: {len(s8_rows)}  S16 rows: {len(s16_rows)}  loaded S8 keys: {len(records)}")

    unresolved = []
    for guide in guides:
        record = find_record_for_label(guide, records)
        s16_row = s16_by_label.get(guide)
        sequence = (s16_row or {}).get("sequence", "")
        gene = (s16_row or {}).get("gene", "") or guide.split("_", 1)[0]
        family = "_".join(guide.split("_")[1:]) if guide.count("_") >= 2 else guide
        candidates = by_gene_seq.get((gene, sequence.upper()), [])
        if record is None:
            unresolved.append(
                {
                    "guide": guide,
                    "gene": gene,
                    "sequence": sequence,
                    "family_prefix": family,
                    "s8_rows_with_same_gene_and_sequence": candidates,
                    "s8_has_family_prefix": family in records,
                    "canonical_keys": canonical_guide_keys(guide),
                }
            )

    print(f"\nunresolved against S8: {len(unresolved)}")
    for item in unresolved[:40]:
        print(
            f"  {item['guide']}\n"
            f"      family={item['family_prefix']}  s8_has_family_key={item['s8_has_family_prefix']}\n"
            f"      s8_rows_by_gene_seq={item['s8_rows_with_same_gene_and_sequence'][:4]}\n"
            f"      canonical={item['canonical_keys']}"
        )

    print("\n--- sample S8 labels for the first unresolved gene family ---")
    if unresolved:
        family = unresolved[0]["family_prefix"]
        head = family.split("_")[0]
        shown = 0
        for row in s8_rows:
            label = str(list(row.values())[0]).strip()
            if label.startswith(head):
                print(f"  label={label!r} gene={row.get('gene')!r} seq={row.get('sgRNA sequence')!r}")
                shown += 1
                if shown >= 8:
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
