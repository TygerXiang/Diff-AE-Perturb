"""Compare two GSE132080 training runs at checkpoint/epoch level."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

RUNS = Path(r"C:\Users\Admin\myprojects\Diff-AE-Perturb\runs")

PAIRS = [
    (
        "historical_legacy_2a1fb63_200ep",
        RUNS / "benchmark_activity_branch_predicted_day10_200ep",
        RUNS / "continuity_check" / "original_2a1fb63_200ep",
    ),
    (
        "historical_explicit_8179140_200ep",
        RUNS / "benchmark_activity_branch_predicted_day10_explicit_only_200ep",
        RUNS / "continuity_check" / "current_worktree_200ep",
    ),
]


def load_ckpt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def compare(name: str, ref_dir: Path, new_dir: Path) -> dict:
    print("=" * 78)
    print(name)
    print("=" * 78)
    ref_ck, new_ck = load_ckpt(ref_dir / "checkpoint_best.pt"), load_ckpt(new_dir / "checkpoint_best.pt")
    print(f"  ref best: epoch={ref_ck['epoch']} val_loss={ref_ck['val_loss']!r}")
    print(f"  new best: epoch={new_ck['epoch']} val_loss={new_ck['val_loss']!r}")
    print(f"  identical perturbation mapping: {ref_ck['perturbation_mapping'] == new_ck['perturbation_mapping']}")

    sr, sn = ref_ck["model_state_dict"], new_ck["model_state_dict"]
    print(f"  same key set: {set(sr) == set(sn)}   n_keys ref/new = {len(sr)}/{len(sn)}")
    print(f"  new arch: activity_condition present = "
          f"{any(k.startswith('activity_condition') for k in sn)}; "
          f"denoiser in-width = {tuple(sn['denoiser.network.0.weight'].shape)}")
    print(f"  ref arch: activity_condition present = "
          f"{any(k.startswith('activity_condition') for k in sr)}; "
          f"denoiser in-width = {tuple(sr['denoiser.network.0.weight'].shape)}")

    if set(sr) == set(sn):
        worst_key, worst, total = None, 0.0, 0.0
        for k in sr:
            d = (sr[k].float() - sn[k].float()).abs().max().item()
            total += d
            if d > worst:
                worst, worst_key = d, k
        print(f"  best-checkpoint max |param diff| = {worst:.3e} (at {worst_key})")
        print(f"  best-checkpoint mean |param diff| = {total / len(sr):.3e}")

    mref = json.loads((ref_dir / "metrics.json").read_text())
    mnew = json.loads((new_dir / "metrics.json").read_text())
    href = {h["epoch"]: h for h in mref["history"]}
    hnew = {h["epoch"]: h for h in mnew["history"]}
    exact = sum(1 for e in href if href[e]["val_loss"] == hnew[e]["val_loss"])
    near = sum(1 for e in href if abs(href[e]["val_loss"] - hnew[e]["val_loss"]) < 1e-6)
    worst_e, worst_d = None, 0.0
    for e in href:
        d = abs(href[e]["val_loss"] - hnew[e]["val_loss"])
        if d > worst_d:
            worst_d, worst_e = d, e
    print(f"  epochs bit-identical val_loss: {exact}/200; within 1e-6: {near}/200")
    print(f"  max epoch val_loss gap: {worst_d:.3e} (epoch {worst_e})")
    print(f"  ref best_epoch/best_val_loss : {mref['best_epoch']} / {mref['best_val_loss']!r}")
    print(f"  new best_epoch/best_val_loss : {mnew['best_epoch']} / {mnew['best_val_loss']!r}")
    print(f"  ref test_metrics : {json.dumps(mref['test_metrics'])}")
    print(f"  new test_metrics : {json.dumps(mnew['test_metrics'])}")
    print()

    return {
        "name": name,
        "ref_dir": str(ref_dir),
        "new_dir": str(new_dir),
        "ref_best_epoch": ref_ck["epoch"],
        "new_best_epoch": new_ck["epoch"],
        "ref_best_val_loss": ref_ck["val_loss"],
        "new_best_val_loss": new_ck["val_loss"],
        "epochs_bit_identical": exact,
        "epochs_within_1e-6": near,
        "max_epoch_val_gap": worst_d,
        "max_epoch_val_gap_epoch": worst_e,
        "ref_test_metrics": mref["test_metrics"],
        "new_test_metrics": mnew["test_metrics"],
        "new_arch": {
            "activity_condition": any(k.startswith("activity_condition") for k in sn),
            "denoiser_in_width": list(sn["denoiser.network.0.weight"].shape),
        },
    }


def main() -> None:
    results = []
    for name, ref_dir, new_dir in PAIRS:
        if not (new_dir / "checkpoint_best.pt").exists():
            print(f"SKIP {name}: {new_dir} has no checkpoint_best.pt yet\n")
            continue
        results.append(compare(name, ref_dir, new_dir))
    out = RUNS / "continuity_check" / "checkpoint_comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
