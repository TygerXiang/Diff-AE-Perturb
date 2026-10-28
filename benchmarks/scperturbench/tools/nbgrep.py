"""Grep notebook code cells for a regex; host-side reconnaissance helper."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("pattern")
    parser.add_argument("--include-annotated", action="store_true")
    args = parser.parse_args()

    rx = re.compile(args.pattern)
    for path in sorted(args.directory.glob("*.ipynb")):
        if not args.include_annotated and "annotated" in path.name:
            continue
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"{path.name}: unreadable ({exc})")
            continue
        for index, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            for line_no, line in enumerate("".join(cell.get("source", [])).splitlines(), start=1):
                if rx.search(line):
                    print(f"{path.name}[{index}:{line_no}] {line.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
