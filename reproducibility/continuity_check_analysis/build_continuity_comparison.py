"""Build the GSE132080 implementation-continuity comparison table and curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\Admin\myprojects\Diff-AE-Perturb")
RUNS = ROOT / "runs"
OUT = RUNS / "continuity_check"

# label -> (metrics.json dir, architecture descriptor, init descriptor)
RUNS_TABLE = [
    (
        "A. original_2a1fb63_historical",
        RUNS / "benchmark_activity_branch_predicted_day10_200ep",
        "legacy: activity scales delta (no explicit channel), denoiser in=1275",
        "scGen latent-mean delta init",
    ),
    (
        "B. original_2a1fb63_reproduced",
        RUNS / "continuity_check" / "original_2a1fb63_200ep",
        "legacy: activity scales delta (no explicit channel), denoiser in=1275",
        "scGen latent-mean delta init",
    ),
    (
        "C. explicit_8179140_historical",
        RUNS / "benchmark_activity_branch_predicted_day10_explicit_only_200ep",
        "explicit 16-d activity channel, denoiser in=1291",
        "scGen latent-mean delta init",
    ),
    (
        "D. explicit_8179140_reproduced",
        RUNS / "continuity_check" / "explicit_8179140_200ep",
        "explicit 16-d activity channel, denoiser in=1291",
        "scGen latent-mean delta init",
    ),
    (
        "E. current_worktree_reproduced",
        RUNS / "continuity_check" / "current_worktree_200ep",
        "explicit 16-d activity channel, denoiser in=1291",
        "no delta init (random embeddings)",
    ),
    (
        "F. original_2a1fb63_cudnn_off",
        RUNS / "continuity_check" / "cudnn_off_deterministic_200ep",
        "legacy: activity scales delta (no explicit channel), denoiser in=1275",
        "scGen latent-mean delta init",
    ),
]

COLORS = ["#1f77b4", "#7fb3d5", "#d62728", "#f0a0a0", "#2ca02c", "#9467bd"]


def load(entry):
    label, path, arch, init = entry
    metrics = json.loads((path / "metrics.json").read_text())
    history = {h["epoch"]: h for h in metrics["history"]}
    return {
        "label": label,
        "path": str(path),
        "arch": arch,
        "init": init,
        "best_epoch": metrics["best_epoch"],
        "best_val_loss": metrics["best_val_loss"],
        "test": metrics["test_metrics"],
        "history": history,
        "epochs": len(metrics["history"]),
    }


def main() -> None:
    rows = []
    for entry in RUNS_TABLE:
        if not (entry[1] / "metrics.json").exists():
            print(f"missing metrics for {entry[0]} - skipped")
            continue
        rows.append(load(entry))

    print("\n=== GSE132080 implementation continuity: summary ===")
    hdr = f"{'run':<34}{'ep':>4}{'best_ep':>8}{'best_val':>12}{'test_loss':>12}{'test_mse':>11}{'test_str':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        t = r["test"]
        print(
            f"{r['label']:<34}{r['epochs']:>4}{r['best_epoch']:>8}"
            f"{r['best_val_loss']:>12.6f}{t['loss']:>12.6f}"
            f"{t['eval_mse_loss']:>11.6f}{t['guide_strength_loss']:>10.6f}"
        )

    # relative deltas against the historical legacy reference
    ref = next((r for r in rows if r["label"].startswith("A.")), None)
    if ref:
        print("\n=== delta vs historical original (A) ===")
        for r in rows:
            if r is ref:
                continue
            dv = r["best_val_loss"] - ref["best_val_loss"]
            dt = r["test"]["loss"] - ref["test"]["loss"]
            print(
                f"{r['label']:<34} d_best_val={dv:+.6f} "
                f"({100 * dv / ref['best_val_loss']:+.4f}%)  "
                f"d_test={dt:+.6f} ({100 * dt / ref['test']['loss']:+.4f}%)"
            )

    # pair-level reproduction fidelity
    print("\n=== reproduction fidelity (historical vs rerun) ===")
    for a, b in (("A.", "B."), ("C.", "D.")):
        ra = next((r for r in rows if r["label"].startswith(a)), None)
        rb = next((r for r in rows if r["label"].startswith(b)), None)
        if not (ra and rb):
            continue
        dv = rb["best_val_loss"] - ra["best_val_loss"]
        dt = rb["test"]["loss"] - ra["test"]["loss"]
        ident = sum(
            1 for e in ra["history"] if ra["history"][e]["val_loss"] == rb["history"][e]["val_loss"]
        )
        near = sum(
            1
            for e in ra["history"]
            if abs(ra["history"][e]["val_loss"] - rb["history"][e]["val_loss"]) < 1e-6
        )
        gap = max(abs(ra["history"][e]["val_loss"] - rb["history"][e]["val_loss"]) for e in ra["history"])
        print(
            f"{a}->{b}  best_ep {ra['best_epoch']}->{rb['best_epoch']}  "
            f"d_best_val={dv:+.3e} ({100 * dv / ra['best_val_loss']:+.5f}%)  "
            f"d_test={dt:+.3e} ({100 * dt / ra['test']['loss']:+.5f}%)  "
            f"bit-identical epochs={ident}/200  within1e-6={near}/200  max_gap={gap:.3e}"
        )

    # architecture-only contrast at matched initialization
    print("\n=== architecture effect (legacy B vs explicit D, both scGen-initialised, same seed) ===")
    b = next((r for r in rows if r["label"].startswith("B.")), None)
    d = next((r for r in rows if r["label"].startswith("D.")), None)
    if b and d:
        print(
            f"best_val {b['best_val_loss']:.6f} -> {d['best_val_loss']:.6f} "
            f"({d['best_val_loss'] - b['best_val_loss']:+.6f})"
        )
        print(
            f"test     {b['test']['loss']:.6f} -> {d['test']['loss']:.6f} "
            f"({d['test']['loss'] - b['test']['loss']:+.6f})"
        )

    # curves
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    for color, r in zip(COLORS, rows):
        eps = sorted(r["history"])
        val = [r["history"][e]["val_loss"] for e in eps]
        mse = [r["history"][e]["val_eval_mse_loss"] for e in eps]
        stg = [r["history"][e]["val_guide_strength_loss"] for e in eps]
        axes[0].plot(eps, val, color=color, lw=1.4, label=r["label"])
        axes[1].plot(eps, mse, color=color, lw=1.4, label=r["label"])
        axes[2].plot(eps, stg, color=color, lw=1.4, label=r["label"])
        axes[0].scatter([r["best_epoch"]], [r["best_val_loss"]], color=color, s=26, zorder=5)

    axes[0].set_title("validation total loss")
    axes[1].set_title("validation eval MSE (diffusion term)")
    axes[2].set_title("validation guide-strength loss")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("GSE132080 200-epoch implementation continuity (seed 7, identical data/split/hparams)")
    fig.tight_layout()
    curve_path = OUT / "continuity_curves.png"
    fig.savefig(curve_path, dpi=130)
    print(f"\nwrote {curve_path}")

    # focus plot: last 60 epochs
    fig2, ax2 = plt.subplots(figsize=(9, 5.2))
    for color, r in zip(COLORS, rows):
        eps = [e for e in sorted(r["history"]) if e >= 140]
        ax2.plot(eps, [r["history"][e]["val_loss"] for e in eps], color=color, lw=1.6, label=r["label"])
        ax2.scatter([r["best_epoch"]], [r["best_val_loss"]], color=color, s=30, zorder=5)
    ax2.set_title("validation total loss, epochs 140-200")
    ax2.set_xlabel("epoch")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    zoom_path = OUT / "continuity_curves_zoom.png"
    fig2.savefig(zoom_path, dpi=130)
    print(f"wrote {zoom_path}")

    # csv
    csv_path = OUT / "continuity_summary.csv"
    lines = ["run,arch,delta_init,epochs,best_epoch,best_val_loss,test_loss,test_eval_mse,test_guide_strength_loss"]
    for r in rows:
        t = r["test"]
        lines.append(
            f'"{r["label"]}","{r["arch"]}","{r["init"]}",{r["epochs"]},{r["best_epoch"]},'
            f'{r["best_val_loss"]!r},{t["loss"]!r},{t["eval_mse_loss"]!r},{t["guide_strength_loss"]!r}'
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")

    # per-epoch csv for all runs
    per_epoch = OUT / "continuity_history.csv"
    plines = ["run,epoch,train_loss,val_loss,val_eval_mse_loss,val_guide_strength_loss"]
    for r in rows:
        for e in sorted(r["history"]):
            h = r["history"][e]
            plines.append(
                f'"{r["label"]}",{e},{h["train_loss"]!r},{h["val_loss"]!r},'
                f'{h["val_eval_mse_loss"]!r},{h["val_guide_strength_loss"]!r}'
            )
    per_epoch.write_text("\n".join(plines) + "\n", encoding="utf-8")
    print(f"wrote {per_epoch}")

    summary = {
        "runs": [
            {
                k: v
                for k, v in r.items()
                if k not in ("history",)
            }
            for r in rows
        ]
    }
    (OUT / "continuity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'continuity_summary.json'}")


if __name__ == "__main__":
    main()
