from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run target-only and dose-conditioned GSE283614 CV training.")
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observed-summary", type=Path, required=True)
    parser.add_argument("--states", nargs="+", default=["ESC", "DE"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Run fold 0, seed 7, and at most 20 epochs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-failed-observed-gate",
        action="store_true",
        help="Override the preregistered stage gate. Intended only for diagnostic smoke runs.",
    )
    return parser.parse_args()


def command_for_run(
    *,
    state: str,
    fold: int,
    seed: int,
    method: str,
    cv_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    device: str,
) -> list[str]:
    split_dir = cv_dir / state.lower() / f"fold_{fold}"
    prefix = f"gse283614_{state.lower()}"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_with_split.py"),
        "--data-path",
        str(split_dir / f"{prefix}_train.h5ad"),
        "--train-path",
        str(split_dir / f"{prefix}_train.h5ad"),
        "--val-path",
        str(split_dir / f"{prefix}_val.h5ad"),
        "--test-path",
        str(split_dir / f"{prefix}_test.h5ad"),
        "--perturbation-col",
        "guide_family",
        "--activity-label-col",
        "guide_identity",
        "--target-strategy",
        "raw",
        "--control-label",
        "control",
        "--prediction-type",
        "x0",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]
    if method == "dose":
        command.extend(
            [
                "--activity-source",
                "direct",
                "--activity-table",
                str(cv_dir / "gse283614_tf_dosage.tsv"),
                "--activity-col",
                "knockdown_fraction",
            ]
        )
    elif method == "target_only":
        command.extend(["--activity-source", "none"])
    else:
        raise ValueError(f"Unknown method: {method}")
    return command


def main() -> None:
    args = parse_args()
    observed = json.loads(args.observed_summary.read_text(encoding="utf-8"))
    gate_pass = bool(observed.get("observed_stage_gate_pass", False))
    if not gate_pass and not args.allow_failed_observed_gate:
        raise RuntimeError(
            "Observed biology stage gate did not pass. Resolve the discrepancies or use "
            "--allow-failed-observed-gate for a diagnostic smoke run only."
        )
    states = [str(value).upper() for value in args.states]
    folds = [0] if args.smoke else args.folds
    seeds = [7] if args.smoke else args.seeds
    epochs = min(args.epochs, 20) if args.smoke else args.epochs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for state in states:
        for fold in folds:
            for seed in seeds:
                for method in ("target_only", "dose"):
                    run_dir = args.output_dir / state.lower() / f"fold_{fold}" / f"seed_{seed}" / method
                    checkpoint = run_dir / "checkpoint_best.pt"
                    command = command_for_run(
                        state=state,
                        fold=fold,
                        seed=seed,
                        method=method,
                        cv_dir=args.cv_dir,
                        output_dir=run_dir,
                        epochs=epochs,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                    record: dict[str, object] = {
                        "state": state,
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "output_dir": str(run_dir),
                        "command": command,
                    }
                    if checkpoint.exists() and not args.force:
                        record["status"] = "skipped_existing"
                    elif args.dry_run:
                        record["status"] = "dry_run"
                    else:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        subprocess.run(command, cwd=REPO_ROOT, check=True)
                        record["status"] = "completed"
                    records.append(record)
                    (args.output_dir / "training_manifest.json").write_text(
                        json.dumps(
                            {
                                "observed_stage_gate_pass": gate_pass,
                                "gate_override": bool(args.allow_failed_observed_gate),
                                "smoke": bool(args.smoke),
                                "epochs": epochs,
                                "runs": records,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
    print(json.dumps({"n_runs": len(records), "manifest": str(args.output_dir / "training_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
