"""Plot the six benchmark metrics for the requested five methods.

The source table exposes full-gene (``@all``) columns for MSE, PCC-delta,
E-distance, and KL divergence. Wasserstein and Common-DEGs retain the
benchmark's protocol-defined 100-gene and DEG-cutoff columns, respectively;
their chart titles intentionally omit the budget suffix.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["linear_ridge", "cpa", "scgen", "gears", "fix_x0_percell"]
LABELS = {
    "linear_ridge": "Linear Ridge",
    "cpa": "CPA",
    "scgen": "scGen",
    "gears": "GEARS",
    "fix_x0_percell": "Diff-AE",
}
COLORS = {
    "linear_ridge": "#2878B5",
    "cpa": "#E07A2D",
    "scgen": "#3A923A",
    "gears": "#C94C4C",
    "fix_x0_percell": "#8064A2",
}
METRICS = [
    ("mse@all", "MSE", "lower is better"),
    ("pcc_delta@all", "PCC-delta", "higher is better"),
    ("e_distance@all", "E-distance", "lower is better"),
    ("kl_divergence@all", "KL divergence", "lower is better"),
    ("wasserstein@100", "Wasserstein", "lower is better"),
    ("common_degs@cutoff", "Common-DEGs", "higher is better"),
]
EXPECTED = {
    ("linear_ridge", "mse@all"): 0.003488,
    ("cpa", "pcc_delta@all"): 0.153094,
    ("scgen", "kl_divergence@all"): 35.495660,
    ("gears", "e_distance@all"): 0.563596,
    ("fix_x0_percell", "mse@all"): 0.005937,
    ("linear_ridge", "wasserstein@100"): 37.535861,
    ("gears", "common_degs@cutoff"): 0.153786,
}


def arguments() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=workspace / ".scperturbench/results/gse132080_6metrics/gse132080/aggregate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs/all_metric_bars",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    source = path / "summary_wide.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing aggregate summary: {source}")
    summary = pd.read_csv(source).set_index("method")
    missing_methods = [method for method in METHODS if method not in summary.index]
    missing_metrics = [metric for metric, _, _ in METRICS if metric not in summary.columns]
    if missing_methods or missing_metrics:
        raise AssertionError(f"Missing methods={missing_methods}; metrics={missing_metrics}")
    if any("squidiff" in method.lower() for method in METHODS):
        raise AssertionError("Unexpected Squidiff method in requested comparison")
    for (method, metric), expected in EXPECTED.items():
        actual = float(summary.loc[method, metric])
        if not np.isclose(actual, expected, rtol=0, atol=1e-6):
            raise AssertionError(f"Unexpected {method}/{metric}: {actual} != {expected}")
    return summary.loc[METHODS]


def format_value(value: float) -> str:
    return f"{value:.6g}"


def draw_metric(
    ax: plt.Axes, values: pd.Series, title: str, direction: str,
    *, show_direction: bool = True,
) -> None:
    y = np.arange(len(METHODS))
    numbers = values.astype(float).to_numpy()
    bars = ax.barh(y, numbers, color=[COLORS[m] for m in METHODS], height=0.68)
    ax.set_yticks(y, [LABELS[m] for m in METHODS])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("Metric value")
    if show_direction:
        ax.text(0.98, 0.04, direction, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    xmax = max(numbers) if len(numbers) else 1.0
    ax.set_xlim(0, xmax * 1.2 if xmax > 0 else 1.0)
    for bar, value in zip(bars, numbers):
        ax.text(
            value + xmax * 0.025,
            bar.get_y() + bar.get_height() / 2,
            format_value(value),
            va="center",
            ha="left",
            fontsize=8,
        )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("svg", {}),
        ("pdf", {}),
    ):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def make_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = "GSE132080 | unseen-guide / seen-target | 15 held-out guides | 1019 genes | seed 7"
    budget_note = "MSE/PCC-delta/E-distance/KL: full gene panel; Wasserstein: 100-gene protocol; Common-DEGs: DEG cutoff"

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (metric, title, direction) in zip(axes.ravel(), METRICS):
        draw_metric(ax, summary[metric], title, direction, show_direction=False)
    fig.suptitle("Six benchmark metrics for perturbation prediction", x=0.01, ha="left", fontsize=15, weight="bold")
    fig.subplots_adjust(top=0.92, hspace=0.36, wspace=0.28, left=0.07, right=0.98, bottom=0.09)
    save(fig, output_dir, "gse132080_full_gene_metric_bars")

    for metric, title, direction in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        draw_metric(ax, summary[metric], title, direction)
        fig.text(0.01, 0.975, metadata, fontsize=8)
        fig.text(0.01, 0.947, budget_note, fontsize=7, color="#555555")
        fig.subplots_adjust(top=0.86, left=0.18, right=0.98, bottom=0.14)
        save(fig, output_dir, f"gse132080_{metric.split('@', 1)[0]}_bars")


def write_manifest(output_dir: Path, aggregate_dir: Path) -> None:
    data = {
        "dataset": "GSE132080",
        "split": "unseen-guide/seen-target",
        "conditions": 15,
        "genes": 1019,
        "seed": 7,
        "source": str(aggregate_dir / "summary_wide.csv"),
        "methods": {method: LABELS[method] for method in METHODS},
        "plotted_source_columns": [metric for metric, _, _ in METRICS],
        "budget_note": "MSE/PCC-delta/E-distance/KL use all genes; Wasserstein uses the protocol-defined 100-gene budget; Common-DEGs uses the protocol-defined DEG cutoff.",
        "uncertainty": "single seed; no error bars",
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = arguments()
    summary = load_summary(args.aggregate_dir)
    make_figures(summary, args.output_dir)
    write_manifest(args.output_dir, args.aggregate_dir)
    print(f"Wrote full-gene metric bar charts to {args.output_dir}")


if __name__ == "__main__":
    main()
