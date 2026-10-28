"""Print raw Table S8 / S16 rows for one gene family to understand the _00 gap."""

from __future__ import annotations

import csv
from pathlib import Path

TABLES = Path("/root/Diff-AE-Perturb/datasets/GSE132080/41587_2019_387_MOESM3_ESM")
S8 = TABLES / "Table_S8_machine_learning_input.txt"
S16 = TABLES / "Table_S16_perturb-seq_screen_phenotypes.txt"
GENE = "ALDOA"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    s8 = rows(S8)
    s16 = rows(S16)

    print(f"=== S16 rows for {GENE} ===")
    for row in s16:
        label = str(list(row.values())[0])
        if label.startswith(GENE + "_"):
            print(f"  {label}  seq={row.get('sequence')}  rel_day10={row.get('relative_activity_day10')}")

    print(f"\n=== S8 rows for {GENE} (first 6) ===")
    count = 0
    total = 0
    for row in s8:
        label = str(list(row.values())[0])
        if label.startswith(GENE + "_"):
            total += 1
            if count < 6:
                print(
                    f"  {label}  gene={row.get('gene')!r} seq={row.get('sgRNA sequence')!r} "
                    f"perfect={row.get('perfect match sgRNA')!r} pos={row.get('mismatch position')!r}"
                )
                count += 1
    print(f"  ... total S8 rows starting with {GENE}_: {total}")

    print("\n=== S8 label suffixes for this gene (distribution) ===")
    suffixes: dict[str, int] = {}
    for row in s8:
        label = str(list(row.values())[0])
        if not label.startswith(GENE + "_"):
            continue
        suffix = label.rsplit("_", 1)[-1]
        suffixes[suffix] = suffixes.get(suffix, 0) + 1
    print(f"  {sorted(suffixes.items(), key=lambda kv: kv[0])}")

    print("\n=== S8 entirely: how many rows end with _00? ===")
    n00 = sum(1 for row in s8 if str(list(row.values())[0]).endswith("_00"))
    print(f"  rows ending _00: {n00} of {len(s8)}")
    print("  first 5 labels:", [str(list(r.values())[0]) for r in s8[:5]])
    print("  last 5 labels:", [str(list(r.values())[0]) for r in s8[-5:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
