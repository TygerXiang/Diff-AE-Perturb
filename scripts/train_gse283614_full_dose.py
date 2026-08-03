from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refit state-specific dose models after successful OOF validation.")
    parser.add_argument("--esc-data", type=Path, required=True)
    parser.add_argument("--de-data", type=Path, required=True)
    parser.add_argument("--dose-table", type=Path, required=True)
    parser.add_argument("--oof-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-failed-oof", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oof = json.loads(args.oof_summary.read_text(encoding="utf-8"))
    if not bool(oof.get("overall_pass", False)) and not args.allow_failed_oof:
        raise RuntimeError("OOF biological validation did not pass; full-data refitting is blocked")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for state, data_path in (("ESC", args.esc_data), ("DE", args.de_data)):
        output = args.output_dir / state.lower()
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_with_split.py"),
            "--data-path",
            str(data_path),
            "--perturbation-col",
            "guide_family",
            "--activity-label-col",
            "guide_identity",
            "--target-strategy",
            "raw",
            "--control-label",
            "control",
            "--activity-source",
            "direct",
            "--activity-table",
            str(args.dose_table),
            "--activity-col",
            "knockdown_fraction",
            "--prediction-type",
            "x0",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--output-dir",
            str(output),
        ]
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        records.append({"state": state, "data": str(data_path), "output": str(output), "command": command})
    manifest = {
        "exploratory_full_refit": True,
        "oof_overall_pass": bool(oof.get("overall_pass", False)),
        "gate_override": bool(args.allow_failed_oof),
        "runs": records,
    }
    (args.output_dir / "full_refit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
