"""Pre-flight check: which ``perturbdiff`` will a *bare* import load, and is it this repo's?

``Diff-AE-Perturb`` and ``diff-project`` both ship an import package called ``perturbdiff``.
The host environment carries an editable install (``__editable__.perturbdiff-0.1.0.pth``) that
points at ``diff-project/src``, so any entry point that does not prepend this repository's
``src`` -- a notebook, a ``python -c``, a test runner -- silently loads a different, older
implementation.  That is a real footgun, and it is invisible unless something checks for it.

This script reports it without importing anything ambiguous:

* it scans ``sys.path`` (including ``.pth`` files) for every ``perturbdiff`` package that a bare
  import could pick up, in resolution order;
* it says whether this repository's copy is the one that would win;
* ``--strict`` exits non-zero when it would not.

Fix the environment with::

    python -m pip uninstall -y perturbdiff
    python -m pip install -e .          # from this repository, if you want it importable
"""

from __future__ import annotations

import argparse
import glob
import os
import site
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = (REPO_ROOT / "src" / "perturbdiff").resolve()


def _effective_path_entries() -> list[Path]:
    """sys.path entries a fresh interpreter would start with, including .pth additions."""
    entries: list[Path] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        if not raw:
            return
        path = Path(raw)
        if str(path) not in seen:
            seen.add(str(path))
            entries.append(path)

    add(str(REPO_ROOT))  # '' == script directory for a bare `python -c`
    for raw in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        add(raw)
    for raw in sys.path:
        add(raw)
    for directory in {*site.getsitepackages(), site.getusersitepackages()}:
        if not isinstance(directory, str) or not os.path.isdir(directory):
            continue
        for pth in glob.glob(os.path.join(directory, "*.pth")):
            try:
                text = Path(pth).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "import " not in line:
                    add(line)
    return entries


def find_candidates() -> list[tuple[Path, str]]:
    """(package dir, provenance) for every importable ``perturbdiff``, in resolution order."""
    found: list[tuple[Path, str]] = []
    for entry in _effective_path_entries():
        candidate = entry / "perturbdiff"
        if (candidate / "__init__.py").is_file():
            provenance = "this repository" if candidate.resolve() == EXPECTED else "competing copy"
            if all(existing[0].resolve() != candidate.resolve() for existing in found):
                found.append((candidate.resolve(), provenance))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when the wrong copy wins")
    args = parser.parse_args()

    candidates = find_candidates()
    print(f"this repository expects: {EXPECTED}")
    if not candidates:
        print("no importable 'perturbdiff' found on the effective sys.path")
        return 1 if args.strict else 0

    print("\nimport candidates in resolution order (first one wins):")
    for index, (candidate, provenance) in enumerate(candidates):
        marker = "  <-- wins" if index == 0 else ""
        print(f"  {index + 1}. [{provenance}] {candidate}{marker}")

    winner, provenance = candidates[0]
    ok = winner == EXPECTED
    print(f"\nbare-import resolution {'is' if ok else 'is NOT'} this repository's copy")
    if not ok:
        print(
            "\nFix one of:\n"
            "  * prepend this repo's src inside the entry point (scripts/ already do), or\n"
            "  * python -m pip uninstall -y perturbdiff\n"
            "    python -m pip install -e .\n"
        )
    return 0 if (ok or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
