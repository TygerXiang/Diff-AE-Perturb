"""A2: is the activity channel actually used, or is the shift sampling noise?

The GSE132080 report could only *recommend* this experiment.  It answers the one
question a null A/B result leaves open: "the extra channel did not help" is a very
different statement from "the extra channel did nothing".

Method (no training, frozen checkpoints):

1. Sample the same control cells for every condition, so the only things that can
   change between runs are the activity value and the sampling noise.
2. Predict held-out guide ``g`` twice: with ``g``'s real activity and with another
   guide's activity from the **same target** (same target embedding, same
   target-specific doser row, so only the scalar differs).
3. Repeat both ``--n-samples`` times with **explicitly identical noise**, and
   compare the noise-averaged predictions.

Why the repetition is not optional
----------------------------------
The first version of this probe ran each condition once and produced shifts of
60-140 in L2 against a between-guide spread of ~0.4 - a ratio of 200x - including
for swaps between two *identical* activity values (0.0 -> 0.0).  That is the
signature of DDIM sampling noise, not of the channel: within a run the model is
scored on its mean prediction (``metrics`` compares ``imputed`` against the mean
of ``stimulated``), so the sampling noise that dominates single draws is the very
thing that averages out downstream.

So the probe reports three quantities per guide:

* ``shift_swapped``  : soundness-preserving activity change, noise-averaged
* ``between_guide``  : spread of noise-averaged predictions *across* guides
* ``noise_floor``    : distance between two noise-averaged predictions of the SAME
  condition with disjoint noise; this is the resolution limit of the measurement

Only ``shift_swapped`` comparable to or above ``noise_floor`` says anything.  A
ratio ``shift / between_guide`` near or below 1 means the channel moves
predictions by less than switching guides does.

Read-only w.r.t. training; writes one CSV per (state, fold, arm).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path("/root/perturbench")
REPO = Path("/root/Diff-AE-Perturb")
for candidate in (WORK, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:  # torch/anndata are only needed to run the probe, not to import it
    import torch
except ImportError:  # pragma: no cover - exercised by the self-test outside the GPU env
    torch = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether the activity channel is used.")
    parser.add_argument("--state", default="esc")
    parser.add_argument("--folds", nargs="+", default=["0"])
    parser.add_argument("--arm", default="activity_on")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-controls", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=8,
                        help="noise draws averaged per condition (for BOTH the real and swapped value)")
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inputs-dir", type=Path,
        default=WORK / ".scperturbench/inputs/gse283614_activity_ablation",
    )
    parser.add_argument(
        "--trained-root", type=Path,
        default=WORK / ".scperturbench/trained/gse283614_activity_ablation",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=WORK / ".scperturbench/results/gse283614_activity_ablation/gse283614",
    )
    return parser.parse_args()


def load_activity(path: Path) -> dict[str, float]:
    import pandas as pd

    frame = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
    return {
        str(row["guide_identity"]): float(row["relative_activity"])
        for row in frame.to_dict("records")
        if row.get("relative_activity") is not None and str(row.get("relative_activity")) != ""
    }


def within_target_swap(
    guide: str, target: str, target_of: dict[str, str], activity: dict[str, float], rng
) -> tuple[str, float]:
    pool = sorted(g for g, t in target_of.items() if t == target and g != guide and g in activity)
    if not pool:
        return guide, activity.get(guide, 0.0)
    other = pool[int(rng.integers(0, len(pool)))]
    return other, float(activity[other])


def main() -> int:
    args = parse_args()
    import importlib.util

    import anndata as ad

    import scperturbench_diffae_loader as loader

    spec = importlib.util.spec_from_file_location(
        "pdb_evaluate_benchmark", REPO / "scripts" / "evaluate_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    conditions = json.loads(
        (args.inputs_dir / f"conditions_{args.state}.json").read_text(encoding="utf-8")
    )
    per_condition = conditions["per_condition"]
    target_of = {
        guide: {"NANOG": "NANOG", "POU5F1": "OCT4"}.get(meta["target_gene"], meta["target_gene"])
        for guide, meta in per_condition.items()
    }
    activity = load_activity(args.inputs_dir / f"gse283614_activity_{args.state}.tsv")

    all_rows: list[dict] = []
    for fold in args.folds:
        split = REPO / "reproducibility/prepared/gse283614/dose_cv" / args.state / f"fold_{fold}"
        test_path = split / f"gse283614_{args.state}_test.h5ad"
        ckpt_dir = args.trained_root / args.state / f"fold_{fold}" / args.arm
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        if not ckpts or not test_path.exists():
            print(f"fold {fold}: nothing to probe in {ckpt_dir}")
            continue
        ckpt = ckpts[-1]

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, mapping, _ = loader.load_model(ckpt, device)
        if getattr(model, "guide_doser", None) is None:
            print(f"fold {fold}: checkpoint has no activity encoder; nothing to probe")
            continue

        test = ad.read_h5ad(test_path)
        labels = test.obs["perturbation_label"].astype(str).to_numpy()
        families = test.obs["guide_family"].astype(str).to_numpy()
        matrix = test.X
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        dense = np.asarray(dense, dtype=np.float32)

        rng = np.random.default_rng(args.seed)
        control_idx = np.flatnonzero(labels == "control")
        if control_idx.size == 0:
            print(f"fold {fold}: no control cells in the test split")
            continue
        take = rng.choice(control_idx, min(args.n_controls, control_idx.size), replace=False)
        control_tensor = torch.from_numpy(dense[np.sort(take)]).to(device)
        n_controls = int(control_tensor.shape[0])
        n_genes = int(control_tensor.shape[1])

        guides = sorted(g for g in set(labels) - {"control"})
        if args.max_conditions:
            guides = guides[: args.max_conditions]

        def predict_many(guide: str, value: float, n_samples: int, noise_seed: int) -> np.ndarray:
            """Noise-averaged prediction for one (guide, activity value)."""

            target_name = families[labels == guide][0]
            if target_name not in mapping:
                raise SystemExit(f"target {target_name!r} not in checkpoint mapping {mapping}")
            target_ids = torch.full((n_controls,), int(mapping[target_name]), dtype=torch.long, device=device)
            features = torch.full((n_controls, 1), float(value), dtype=torch.float32, device=device)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(noise_seed))
            accum = torch.zeros((n_controls, n_genes), dtype=torch.float32, device=device)
            for _ in range(n_samples):
                noise = torch.randn(
                    (n_controls, n_genes), generator=generator, device=device, dtype=torch.float32
                )
                accum += model.sample_counterfactual(
                    x_ctrl=control_tensor,
                    target_ids=target_ids,
                    guide_features=features,
                    steps=args.ddim_steps,
                    noise=noise,
                )
            return (accum / n_samples).detach().cpu().numpy().astype(np.float64)

        # Noise-averaged mean prediction per guide, using the REAL activity.
        true_means = {
            guide: predict_many(guide, float(activity.get(guide, 0.0)), args.n_samples, 1000 + 17 * i)
            for i, guide in enumerate(guides)
        }
        stacked = np.stack([true_means[g] for g in guides])
        between_guide = float(np.linalg.norm(stacked - stacked.mean(axis=0, keepdims=True), axis=1).mean())

        # Resolution limit: same condition, disjoint noise draws.
        floor_rows = []
        for i, guide in enumerate(guides[: min(3, len(guides))]):
            a = predict_many(guide, float(activity.get(guide, 0.0)), args.n_samples, 500_000 + i)
            b = predict_many(guide, float(activity.get(guide, 0.0)), args.n_samples, 900_000 + i)
            floor_rows.append(float(np.linalg.norm(a - b)))
        noise_floor = float(np.mean(floor_rows)) if floor_rows else float("nan")

        for i, guide in enumerate(guides):
            true_value = float(activity.get(guide, 0.0))
            swapped_guide, swapped_value = within_target_swap(
                guide, target_of[guide], target_of, activity, rng
            )
            swapped_mean = predict_many(guide, swapped_value, args.n_samples, 1000 + 17 * i)
            shift = float(np.linalg.norm(swapped_mean - true_means[guide]))
            all_rows.append(
                {
                    "state": args.state,
                    "fold": int(fold),
                    "arm": args.arm,
                    "checkpoint": ckpt.name,
                    "guide": guide,
                    "target": target_of[guide],
                    "activity_true": true_value,
                    "activity_swapped": swapped_value,
                    "activity_swap_from": swapped_guide,
                    "activity_delta": swapped_value - true_value,
                    "l2_shift_swapped": shift,
                    "l2_between_guide": between_guide,
                    "l2_noise_floor": noise_floor,
                    "ratio_shift_over_between_guide": shift / between_guide if between_guide else float("nan"),
                    "ratio_shift_over_noise_floor": shift / noise_floor if noise_floor else float("nan"),
                    "n_controls": n_controls,
                    "n_samples": int(args.n_samples),
                    "ddim_steps": int(args.ddim_steps),
                }
            )
            print(
                f"  {guide}: act {true_value:.4f} -> {swapped_value:.4f} ({swapped_guide}) | "
                f"shift={shift:.4f} between-guide={between_guide:.4f} noise-floor={noise_floor:.4f} | "
                f"shift/between={shift / between_guide if between_guide else float('nan'):.3f} "
                f"shift/floor={shift / noise_floor if noise_floor else float('nan'):.3f}"
            )

        out_dir = args.output_dir / args.state / f"fold_{fold}" / args.arm
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"activity_perturbation_probe_seed{args.seed}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows([r for r in all_rows if r["fold"] == int(fold)])
        ratios = np.array(
            [r["ratio_shift_over_between_guide"] for r in all_rows if r["fold"] == int(fold)], dtype=float
        )
        print(
            f"fold {fold}: wrote {out_path} | median shift/between={np.nanmedian(ratios):.4f} "
            f"max={np.nanmax(ratios):.4f} n={ratios.size}"
        )

    if all_rows:
        ratios = np.array([r["ratio_shift_over_between_guide"] for r in all_rows], dtype=float)
        floors = np.array([r["ratio_shift_over_noise_floor"] for r in all_rows], dtype=float)
        print(
            f"\nALL FOLDS: n={len(all_rows)} median shift/between={np.nanmedian(ratios):.4f} "
            f"median shift/noise-floor={np.nanmedian(floors):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
