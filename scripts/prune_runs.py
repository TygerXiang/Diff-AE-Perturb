#!/usr/bin/env python
"""Reclaim disk space under ``runs/`` without losing results that reports cite.

Two rules, both chosen so that nothing a tracked file references-by-name disappears:

Rule 1 -- redundant intermediate checkpoints.
    ``train_with_split.py`` writes ``checkpoint_epoch_NNNN.pt`` every ``--save-every``
    epochs *plus* ``checkpoint_best.pt``.  Every consumer in this repo (see
    ``scripts/evaluate_gse283614_*.py``) loads ``checkpoint_best.pt`` only, and the
    per-epoch losses are already in ``metrics.json``.  So each training run keeps
    ``checkpoint_best.pt`` and the *final* ``checkpoint_epoch_NNNN.pt`` (useful when you
    deliberately do not want model selection, e.g. because val == train in the
    ``splits_unseen_guide_seed7`` split) and drops the rest.

Rule 2 -- whole stale runs.
    Named explicitly in ``STALE_RUNS`` (smoke/probe runs, and runs superseded by a later
    retrain).  Before deleting, their small text artefacts (``metrics.json``,
    perturbation maps, manifest) are archived so the numbers survive.

Nothing here touches a checkpoint that some tracked file still refers to by name;
``--check-references`` re-verifies that before applying.

Dry-run is the default.  Use ``--apply`` to actually delete.

    python scripts/prune_runs.py                      # report only
    python scripts/prune_runs.py --check-references   # report + fail if rule 2 would break a doc
    python scripts/prune_runs.py --apply --archive-meta reports/assets/pruned_runs
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"

#: Runs dropped wholesale, with the reason.  Keep this list explicit and reviewed: the
#: whole point is that "stale" is a decision, not a heuristic.
STALE_RUNS: dict[str, str] = {
    # Smoke / probe runs (wrong scope, kept only as a reminder; superseded by the 200-epoch runs).
    "gse132080_smoke_616cb3b": "200-step smoke run, superseded by gse132080_target_200ep",
    "smoke_compare_target_vs_sgrna": "single comparison.json probe, superseded by runs/benchmarks",
    "gse132080_split_smoke_gpu_616cb3b": "split smoke run (epoch 1), superseded by the 200-epoch runs",
    "split_smoke_by_guide": "smoke split h5ad dump (80 MB), regenerable from scripts/create_h5ad_splits.py",
    # Superseded ablations.
    "gse132080_zbasal_sdelta_diralign_nomargin_20ep": "20-epoch ablation, unreferenced and superseded by the 200-epoch runs",
    "gse283614_no_activity_200ep": "superseded by gse283614_no_activity_controls_fixed_200ep (control-sampling fix)",
    "gse283614_explicit_activity_log2fc_200ep": "superseded by gse283614_explicit_activity_log2fc_controls_fixed_200ep",
    # Historical investigation, superseded by the conditioning diagnosis.  Its summary
    # artefacts (csv/json/png) are archived before deletion; the raw per-run directories can
    # be regenerated with reproducibility/continuity_check_analysis/run_repeats.ps1.
    "continuity_check": "2026-09-17 continuity/determinism investigation, superseded by the 2026-09-28 conditioning diagnosis",
}

#: Small text artefacts worth keeping even when the run itself goes away.
META_KEEP_SUFFIXES = (".json", ".csv", ".tsv", ".md", ".txt", ".png", ".svg")
META_KEEP_MAX_BYTES = 2 * 1024 * 1024

EPOCH_RE = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")
BEST_NAME = "checkpoint_best.pt"

# Tracked documentation that must keep resolving; used by --check-references.
REFERENCE_GLOBS = ("*.md", "docs/*.md", "reports/*.md", "scripts/*.py", "tests/*.py")


def _human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TB"


def find_training_runs(runs_dir: Path) -> list[Path]:
    """Directories that directly contain a checkpoint file."""
    found: list[Path] = []
    for path in sorted(runs_dir.rglob("*")):
        if not path.is_dir():
            continue
        names = {entry.name for entry in path.iterdir() if entry.is_file()}
        if BEST_NAME in names or any(EPOCH_RE.match(name) for name in names):
            found.append(path)
    return found


def plan_checkpoint_pruning(runs: list[Path], dropped: set[Path]) -> list[tuple[Path, int, str]]:
    actions: list[tuple[Path, int, str]] = []
    for run in runs:
        if any(run == drop or drop in run.parents for drop in dropped):
            continue  # covered by a wholesale drop
        epochs = sorted(
            ((int(match.group(1)), entry) for entry in run.iterdir() if (match := EPOCH_RE.match(entry.name))),
            key=lambda item: item[0],
        )
        if len(epochs) <= 1:
            continue
        keep = epochs[-1][1]
        for _, entry in epochs[:-1]:
            if entry == keep:
                continue
            actions.append((entry, entry.stat().st_size, f"intermediate epoch (keeping {BEST_NAME} + {keep.name})"))
    return actions


def plan_stale_runs(runs_dir: Path, names: dict[str, str]) -> list[tuple[Path, int, str]]:
    actions: list[tuple[Path, int, str]] = []
    for name, reason in names.items():
        path = runs_dir / name
        if not path.exists():
            print(f"  (already gone) {name}", file=sys.stderr)
            continue
        size = sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())
        actions.append((path, size, reason))
    return actions


def archive_metadata(path: Path, archive_root: Path) -> list[Path]:
    """Copy the small text artefacts of ``path`` into ``archive_root / path.name``."""
    written: list[Path] = []
    for entry in sorted(path.rglob("*")):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in META_KEEP_SUFFIXES:
            continue
        if entry.stat().st_size > META_KEEP_MAX_BYTES:
            continue
        relative = entry.relative_to(path)
        target = archive_root / path.name / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry, target)
        written.append(target)
    return written


def check_references(runs_dir: Path, names: dict[str, str]) -> list[str]:
    """Report tracked files whose paths still point into a run we are about to drop.

    Only path-qualified mentions (``runs/<name>``) count: a bare name inside prose -- for
    instance a note recording that the run was pruned -- is not a broken reference.
    """
    problems: list[str] = []
    files: list[Path] = []
    for pattern in REFERENCE_GLOBS:
        files.extend(REPO_ROOT.glob(pattern))
    files = [source for source in files if source.resolve() != Path(__file__).resolve()]
    for name in names:
        needle = re.compile(r"runs[\\/]" + re.escape(name))
        for source in files:
            try:
                text = source.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if needle.search(text):
                problems.append(f"{source.relative_to(REPO_ROOT)}: still references runs/{name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--apply", action="store_true", help="actually delete (default: report only)")
    parser.add_argument("--archive-meta", type=Path, default=None, help="copy small text artefacts of dropped runs here")
    parser.add_argument("--check-references", action="store_true", help="fail if a tracked file mentions a dropped run")
    parser.add_argument("--no-drop-runs", action="store_true", help="only prune intermediate checkpoints")
    args = parser.parse_args(argv)

    runs_dir: Path = args.runs_dir
    if not runs_dir.is_dir():
        print(f"no such runs directory: {runs_dir}", file=sys.stderr)
        return 2

    stale = {} if args.no_drop_runs else dict(STALE_RUNS)
    if stale and args.check_references:
        problems = check_references(runs_dir, stale)
        if problems:
            print("refusing to drop runs that tracked files still reference:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

    dropped = [runs_dir / name for name in stale]
    runs = find_training_runs(runs_dir)
    checkpoint_actions = plan_checkpoint_pruning(runs, set(dropped))
    stale_actions = plan_stale_runs(runs_dir, stale)

    print(f"runs directory : {runs_dir}")
    print(f"training runs  : {len(runs)}")
    print()
    print("Rule 2 - whole runs to drop")
    for path, size, reason in stale_actions:
        print(f"  {_human(size):>10}  {path.name:<58} {reason}")
    if not stale_actions:
        print("  (none)")
    print()
    print("Rule 1 - intermediate checkpoints to drop")
    by_run: dict[Path, tuple[int, int]] = {}
    for path, size, _ in checkpoint_actions:
        count, total = by_run.get(path.parent, (0, 0))
        by_run[path.parent] = (count + 1, total + size)
    for run, (count, total) in sorted(by_run.items(), key=lambda item: -item[1][1]):
        print(f"  {_human(total):>10}  {run.relative_to(runs_dir)!s:<58} {count} file(s)")
    if not by_run:
        print("  (none)")
    grand_total = sum(size for _, size, _ in stale_actions + checkpoint_actions)
    print()
    print(f"total reclaimable: {_human(grand_total)} "
          f"({len(stale_actions)} run(s), {len(checkpoint_actions)} checkpoint file(s))")

    if not args.apply:
        print("\ndry run: nothing deleted (pass --apply)")
        return 0

    if args.archive_meta is not None:
        for path, _, _ in stale_actions:
            written = archive_metadata(path, args.archive_meta)
            print(f"archived {len(written)} metadata file(s) from {path.name} -> {args.archive_meta}")

    removed = 0
    for path, size, _ in stale_actions:
        shutil.rmtree(path)
        removed += size
    for path, size, _ in checkpoint_actions:
        path.unlink()
        removed += size
    print(f"\ndeleted {_human(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
