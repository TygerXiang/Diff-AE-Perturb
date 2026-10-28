"""Dump the source cells of a Jupyter notebook as plain text.

Host-side helper only: this never participates in benchmark scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--cell", type=int, default=None, help="print only this cell index")
    args = parser.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(nb.get("cells", [])):
        kind = cell.get("cell_type", "?")
        if args.code_only and kind != "code":
            continue
        if args.cell is not None and index != args.cell:
            continue
        print(f"===== [{index}] {kind} =====")
        print("".join(cell.get("source", [])))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
