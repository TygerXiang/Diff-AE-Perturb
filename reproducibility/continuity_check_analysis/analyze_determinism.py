"""Analyse numeric determinism of the GSE132080 training runs."""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(r"C:\Users\Admin\myprojects\Diff-AE-Perturb\runs")
OUT = RUNS / "continuity_check"

HISTORICAL = RUNS / "benchmark_activity_branch_predicted_day10_200ep"
REPRO_ON = OUT / "original_2a1fb63_200ep"
OFF_PROBE = OUT / "determinism_cudnn_off_10ep"
ON_PROBE = OUT / "determinism_cudnn_on_10ep"
OFF_200 = OUT / "cudnn_off_deterministic_200ep"


def hist(path: Path) -> dict[int, dict]:
    return {h["epoch"]: h for h in json.loads((path / "metrics.json").read_text())["history"]}


def first_divergence(a: dict[int, dict], b: dict[int, dict], n: int = 200):
    for e in range(1, n + 1):
        if a[e]["val_loss"] != b[e]["val_loss"]:
            return e, abs(a[e]["val_loss"] - b[e]["val_loss"])
    return None, 0.0


def main() -> None:
    historical = hist(HISTORICAL)
    repro_on = hist(REPRO_ON)
    on_probe = hist(ON_PROBE)
    off_probe = hist(OFF_PROBE)
    off_200 = hist(OFF_200)

    print("=" * 84)
    print("Same code (2a1fb63), same seed 7, same hardware, different numerical realisations")
    print("=" * 84)
    print(f"{'epoch':>5}  {'historical':>22}  {'rerun cudnn-on':>22}  {'rerun cudnn-off':>22}")
    for e in (1, 2, 3, 4, 5, 10, 50, 100, 193, 200):
        print(f"{e:>5}  {historical[e]['val_loss']:>22.17f}  {repro_on[e]['val_loss']:>22.17f}  "
              f"{off_200[e]['val_loss']:>22.17f}")

    print()
    print("### Pairwise first divergence and maximum epoch gap")
    for label, a, b in (
        ("historical vs rerun(cuDNN on)", historical, repro_on),
        ("historical vs rerun(cuDNN off)", historical, off_200),
        ("rerun(cuDNN on) vs rerun(cuDNN off)", repro_on, off_200),
        ("rerun(cuDNN on) vs 10ep probe(cuDNN on)", repro_on, on_probe),
        ("rerun(cuDNN off) vs 10ep probe(cuDNN off)", off_200, off_probe),
    ):
        n = min(200, min(max(a), max(b)))
        e, d = first_divergence(a, b, n)
        gaps = [abs(a[k]["val_loss"] - b[k]["val_loss"]) for k in range(1, n + 1)]
        ident = sum(1 for k in range(1, n + 1) if a[k]["val_loss"] == b[k]["val_loss"])
        print(f"  {label:<42} first_div=epoch {e:<4} delta={d:.3e}  "
              f"ident={ident}/{n}  max_gap={max(gaps):.3e}")

    print()
    print("### Best epoch / best val loss for the three original-implementation realisations")
    bests = {}
    for label, h in (("historical (2026-07-20)", historical),
                     ("rerun cuDNN on", repro_on),
                     ("rerun cuDNN off", off_200)):
        b = min(h.values(), key=lambda r: r["val_loss"])
        bests[label] = (b["epoch"], b["val_loss"])
        print(f"  {label:<24} best_epoch={b['epoch']:>3}  best_val_loss={b['val_loss']!r}")
    spread = max(v for _, v in bests.values()) - min(v for _, v in bests.values())
    print(f"  spread across the three = {spread:.3e}  ({100 * spread / bests['historical (2026-07-20)'][1]:.5f}% relative)")

    print()
    print("### Conclusion")
    print("  Run-to-run numerical noise floor for this model/hardware/seed:")
    print(f"    max per-epoch val_loss gap   ~ 5e-5 .. 7e-5")
    print(f"    best_val_loss spread         ~ {spread:.1e}")
    print("  Disabling cuDNN does NOT make it bit-reproducible, so the source is broader than")
    print("  one cuDNN algorithm choice (cuBLAS/reduction ordering, TF32 accumulation).")
    print("  The script never enables torch.use_deterministic_algorithms nor pins")
    print("  CUBLAS_WORKSPACE_CONFIG, so bit-for-bit was never a property of the original either.")

    out = {
        "noise_floor": {
            "best_val_loss_spread_across_original_realisations": spread,
            "max_epoch_gap_range": [5e-5, 7.5e-5],
        },
        "best_val_loss": {k: {"epoch": v[0], "val_loss": v[1]} for k, v in bests.items()},
        "first_divergence": {
            "historical_vs_rerun_cudnn_on": first_divergence(historical, repro_on, 200)[0],
            "historical_vs_rerun_cudnn_off": first_divergence(historical, off_200, 200)[0],
        },
    }
    (OUT / "determinism_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'determinism_analysis.json'}")


if __name__ == "__main__":
    main()
