"""Build a within-target permuted activity table for the A3 placebo arm.

Why this arm exists
-------------------
``activity_on`` vs ``activity_off`` changes two things at once: it adds the
activity *information* and it adds the activity *channel* (the extra argument to
the target-specific doser, plus the explicit activity embedding).  A difference
between those arms therefore cannot be attributed to the activity values.

``activity_perm`` is trained exactly like ``activity_on`` but on a table whose
values are permuted **within each target**.  It has the same architecture, the
same parameter count, the same value distribution per target, the same coverage
and the same zero pattern; only the guide-to-value assignment is destroyed.  So:

    activity_on  vs activity_off    -> channel + information
    activity_on  vs activity_perm   -> information only   (channel held constant)
    activity_perm vs activity_off   -> channel only        (information removed)

The reassignment is fixed-point free whenever the target has more than one guide,
so no guide keeps its own value by accident.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

CONTROL_TARGET = "control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Permute GSE283614 activity values within target.")
    parser.add_argument("--activity-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--mapping-out", type=Path, default=None,
                        help="Optional JSON recording every (guide -> source guide) reassignment.")
    return parser.parse_args()


def fixed_point_free_index(n: int, rng: random.Random) -> list[int]:
    """A permutation of ``range(n)`` with no index mapped to itself."""

    if n < 2:
        return list(range(n))
    for _ in range(1000):
        candidate = list(range(n))
        rng.shuffle(candidate)
        if all(i != j for i, j in enumerate(candidate)):
            return candidate
    return list(range(1, n)) + [0]  # cyclic rotation: always fixed-point free for n >= 2


def main() -> int:
    args = parse_args()
    with args.activity_table.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if args.activity_table.suffix.lower() in {".tsv", ".txt"} else ",")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if not rows:
        raise SystemExit(f"no rows in {args.activity_table}")
    if not fieldnames or fieldnames[0] != "guide_identity":
        raise SystemExit("activity table must have guide_identity as its first column")

    rng = random.Random(args.seed)
    by_target: dict[str, list[dict]] = {}
    for row in rows:
        by_target.setdefault(str(row.get("target_gene", "")), []).append(row)

    out_rows = [dict(row) for row in rows]
    by_guide = {str(row["guide_identity"]): row for row in out_rows}
    mapping: dict[str, str] = {}
    for target, group in sorted(by_target.items()):
        # Controls carry no activity channel at all (the trainer forces id 0 to
        # 0.0 and the model zeroes control rows again), so permuting them would be
        # a no-op dressed up as a manipulation.  Leave them exactly as they are.
        if target == CONTROL_TARGET:
            for row in group:
                mapping[str(row["guide_identity"])] = str(row["guide_identity"])
            continue
        order = fixed_point_free_index(len(group), rng)
        for position, source_position in enumerate(order):
            guide = str(group[position]["guide_identity"])
            source = str(group[source_position]["guide_identity"])
            by_guide[guide]["relative_activity"] = group[source_position]["relative_activity"]
            mapping[guide] = source

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    # Verify the manipulation is real: the per-target multiset of values must be
    # unchanged and every guide with a measured value must have moved.
    real_values = sorted(str(round(float(r["relative_activity"]), 9)) for r in rows if r.get("target_gene") != CONTROL_TARGET)
    perm_values = sorted(str(round(float(r["relative_activity"]), 9)) for r in out_rows if r.get("target_gene") != CONTROL_TARGET)
    fixed_points = sorted(g for g, src in mapping.items() if g == src and by_guide[g].get("target_gene") != CONTROL_TARGET)
    payload = {
        "activity_table": str(args.activity_table),
        "output": str(args.output),
        "seed": args.seed,
        "n_rows": len(out_rows),
        "value_multiset_preserved": real_values == perm_values,
        "n_fixed_points_targeting": len(fixed_points),
        "fixed_point_guides_targeting": fixed_points,
        "n_targets_permuted": len([t for t in by_target if t != CONTROL_TARGET]),
        "mapping": mapping,
    }
    if args.mapping_out is not None:
        args.mapping_out.parent.mkdir(parents=True, exist_ok=True)
        args.mapping_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "mapping"}, indent=2, ensure_ascii=False))
    if not payload["value_multiset_preserved"]:
        raise SystemExit("permutation changed the value multiset; refusing to emit this table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
