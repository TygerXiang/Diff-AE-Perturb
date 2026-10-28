from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Run the current Diff-AE-Perturb evaluator with provenance checks.")
    p.add_argument("--repo", type=Path, required=True, help="Current Diff-AE-Perturb repository root.")
    p.add_argument("--train-path", type=Path, required=True)
    p.add_argument("--test-path", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-prefix", default="diff_ae_current")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-level", choices=["target", "guide"], default="guide")
    p.add_argument("--target-strategy", choices=["first_token", "raw"], default="first_token")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--min-test-cells-per-condition", type=int, default=20)
    p.add_argument("--max-control-cells", type=int, default=128)
    p.add_argument("--max-metric-cells", type=int, default=2000)
    p.add_argument("--control-label", action="append", default=["control"])
    args = p.parse_args()

    repo = args.repo.resolve()
    evaluator = repo / "scripts" / "evaluate_benchmark.py"
    package_dir = repo / "src" / "perturbdiff"
    if not evaluator.exists() or not package_dir.exists():
        raise SystemExit(f"Not a Diff-AE-Perturb checkout: {repo}")
    for label, path in (("repo", repo), ("checkpoint", args.checkpoint)):
        if "diff-project" in str(path).lower() and "Diff-AE-Perturb" not in str(path):
            raise SystemExit(f"Refusing stale diff-project path for {label}: {path}")
    for label, path in (("train", args.train_path), ("test", args.test_path), ("checkpoint", args.checkpoint)):
        if not path.exists():
            raise SystemExit(f"Missing {label} path: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.output_prefix}_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Diff-AE-Perturb",
        "repo": str(repo),
        "repo_evaluator": str(evaluator),
        "repo_evaluator_sha256": sha256(evaluator),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "train_path": str(args.train_path.resolve()),
        "test_path": str(args.test_path.resolve()),
        "device": args.device,
        "eval_level": args.eval_level,
        "target_strategy": args.target_strategy,
        "seed": 7,
        "status": "running",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(evaluator),
        "--train-path", str(args.train_path),
        "--test-path", str(args.test_path),
        "--perturbdiff-checkpoint", str(args.checkpoint),
        "--perturbdiff-name", "diff_ae_perturb_current",
        "--output-dir", str(args.output_dir),
        "--output-prefix", args.output_prefix,
        "--device", args.device,
        "--eval-level", args.eval_level,
        "--target-strategy", args.target_strategy,
        "--ddim-steps", str(args.ddim_steps),
        "--min-test-cells-per-condition", str(args.min_test_cells_per_condition),
        "--max-control-cells", str(args.max_control_cells),
        "--max-metric-cells", str(args.max_metric_cells),
    ]
    for label in args.control_label:
        cmd.extend(["--control-label", label])

    log_path = args.output_dir / f"{args.output_prefix}.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)

    manifest["status"] = "ok" if proc.returncode == 0 else "failed"
    manifest["returncode"] = proc.returncode
    manifest["command"] = cmd
    manifest["log"] = str(log_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
