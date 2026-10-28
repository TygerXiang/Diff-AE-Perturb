"""Create reproducible GSE132080 benchmark comparison figures.

The script consumes only the frozen aggregate CSV files produced by the
benchmark runner. It intentionally excludes all Squidiff and failed eps
sampling variants from the displayed method set.
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
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch


METHOD_ORDER = [
    "control_mean",
    "train_mean",
    "linear_pca",
    "linear_ridge",
    "cpa",
    "scgen",
    "gears",
    "diffae",
    "diffae_delta_recompute",
    "fix_x0_percell",
    "fix_x0_batchmean",
]

METHOD_LABELS = {
    "control_mean": "Control mean",
    "train_mean": "Train mean",
    "linear_pca": "Linear PCA",
    "linear_ridge": "Linear Ridge",
    "cpa": "CPA",
    "scgen": "scGen",
    "gears": "GEARS",
    "diffae": "Diff-AE",
    "diffae_delta_recompute": "Diff-AE delta recompute",
    "fix_x0_percell": "Diff-AE fix x0 per-cell",
    "fix_x0_batchmean": "Diff-AE fix x0 batch-mean",
}

METHOD_GROUPS = {
    "control_mean": "reference",
    "train_mean": "reference",
    "linear_pca": "linear",
    "linear_ridge": "linear",
    "cpa": "deep",
    "scgen": "deep",
    "gears": "deep",
    "diffae": "diffae",
    "diffae_delta_recompute": "repair",
    "fix_x0_percell": "repair",
    "fix_x0_batchmean": "repair",
}

GROUP_COLORS = {
    "reference": "#8c8c8c",
    "linear": "#2166ac",
    "deep": "#d6604d",
    "diffae": "#1b9e77",
    "repair": "#7b3294",
}

METRIC_SPECS = [
    ("mse@100", "MSE @100", "lower"),
    ("mse@all", "MSE @all", "lower"),
    ("pcc_delta@100", "PCC-delta @100", "higher"),
    ("pcc_delta@all", "PCC-delta @all", "higher"),
    ("e_distance@100", "E-distance @100", "lower"),
    ("e_distance@all", "E-distance @all", "lower"),
    ("wasserstein@100", "Wasserstein @100", "lower"),
    ("kl_divergence@100", "KL divergence @100", "lower"),
    ("kl_divergence@all", "KL divergence @all", "lower"),
    ("common_degs@cutoff", "Common-DEGs", "higher"),
]

EXPECTED_VALUES = {
    ("linear_ridge", "mse@100"): 0.024521,
    ("linear_ridge", "pcc_delta@100"): 0.867429,
    ("linear_ridge", "e_distance@100"): 0.513460,
    ("scgen", "mse@all"): 0.002984,
    ("gears", "pcc_delta@100"): 0.895589,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    default_aggregate = root / ".scperturbench" / "results" / "gse132080_6metrics" / "gse132080" / "aggregate"
    default_output = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=default_aggregate)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def load_data(aggregate_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = aggregate_dir / "summary_wide.csv"
    rankings_path = aggregate_dir / "rankings.csv"
    wins_path = aggregate_dir / "wins.csv"
    for path in (summary_path, rankings_path, wins_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark artifact: {path}")
    summary = pd.read_csv(summary_path).set_index("method")
    rankings = pd.read_csv(rankings_path)
    wins = pd.read_csv(wins_path)
    return summary, rankings, wins


def validate_data(summary: pd.DataFrame, rankings: pd.DataFrame, wins: pd.DataFrame) -> pd.DataFrame:
    missing_methods = [method for method in METHOD_ORDER if method not in summary.index]
    if missing_methods:
        raise AssertionError(f"Missing selected methods: {missing_methods}")
    displayed = summary.loc[METHOD_ORDER].copy()
    if any("squidiff" in method.lower() for method in displayed.index):
        raise AssertionError("Squidiff must not appear in the displayed method set")

    for (method, metric), expected in EXPECTED_VALUES.items():
        actual = float(summary.loc[method, metric])
        if not np.isclose(actual, expected, rtol=0, atol=1e-6):
            raise AssertionError(f"Unexpected {method} {metric}: {actual} != {expected}")

    rank_wins = rankings.loc[rankings["rank"].eq(1), "method"].value_counts().sort_index()
    csv_wins = wins.set_index("method")["wins"].astype(int).sort_index()
    if not rank_wins.equals(csv_wins):
        raise AssertionError(
            "wins.csv disagrees with rankings.csv: "
            f"recomputed={rank_wins.to_dict()} csv={csv_wins.to_dict()}"
        )
    return displayed


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rank_heatmap(rankings: pd.DataFrame, output_dir: Path) -> None:
    metric_order = [metric for metric, _, _ in METRIC_SPECS]
    rank_table = (
        rankings[rankings["method"].isin(METHOD_ORDER) & rankings["metric_budget"].isin(metric_order)]
        .pivot(index="method", columns="metric_budget", values="rank")
        .reindex(index=METHOD_ORDER, columns=metric_order)
    )
    if rank_table.isna().any().any():
        raise AssertionError("Rankings are incomplete for the selected methods")

    fig, ax = plt.subplots(figsize=(14, 7.5))
    max_rank = int(rankings["rank"].max())
    bounds = np.arange(0.5, max_rank + 1.5, 1)
    image = ax.imshow(rank_table.to_numpy(), cmap="RdYlGn_r", norm=BoundaryNorm(bounds, 256), aspect="auto")
    ax.set_xticks(range(len(metric_order)), [label for _, label, _ in METRIC_SPECS], rotation=35, ha="right")
    ax.set_yticks(range(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_xlabel("Metric and evaluation budget")
    for row in range(rank_table.shape[0]):
        for col in range(rank_table.shape[1]):
            rank = int(rank_table.iat[row, col])
            ax.text(col, row, str(rank), ha="center", va="center", fontsize=8, color="black")
    ax.set_xticks(np.arange(-0.5, len(metric_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar_ticks = [tick for tick in (1, 5, 10, 15, max_rank) if tick <= max_rank]
    cbar = fig.colorbar(image, ax=ax, ticks=cbar_ticks, pad=0.02)
    cbar.set_label("Rank among evaluated methods")
    fig.suptitle("GSE132080 benchmark ranks (lower rank is better)", x=0.01, ha="left", y=0.99, weight="bold")
    fig.text(0.01, 0.95, "Unseen guide / seen target | 15 held-out conditions | 1019 genes | seed 7", fontsize=9)
    fig.text(0.01, 0.01, "Squidiff variants and failed eps-sampling variants are excluded by design.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    save_figure(fig, output_dir, "gse132080_rank_heatmap")


def plot_baseline_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(16, 13), constrained_layout=True)
    axes = axes.ravel()
    y_positions = np.arange(len(METHOD_ORDER))[::-1]
    for ax, (metric, title, direction) in zip(axes, METRIC_SPECS):
        values = summary.loc[METHOD_ORDER, metric].astype(float).to_numpy()
        for y, method, value in zip(y_positions, METHOD_ORDER, values):
            color = GROUP_COLORS[METHOD_GROUPS[method]]
            marker = "D" if method == "linear_ridge" else "o"
            size = 42 if method == "linear_ridge" else 28
            ax.scatter(value, y, s=size, color=color, edgecolor="black" if method == "linear_ridge" else "white", linewidth=0.8, marker=marker, zorder=3)
        ridge_value = float(summary.loc["linear_ridge", metric])
        ax.axvline(ridge_value, color="#222222", linestyle="--", linewidth=0.9, zorder=1)
        ax.set_yticks(y_positions, [METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=7)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.tick_params(axis="x", labelsize=8)
        ax.text(0.98, 0.04, "lower is better" if direction == "lower" else "higher is better", transform=ax.transAxes, ha="right", fontsize=7, color="#555555")
        ax.set_axisbelow(True)
    fig.suptitle("Deep-learning methods versus the linear Ridge baseline", x=0.02, ha="left", fontsize=15, weight="bold")
    fig.text(0.02, 0.955, "Raw benchmark values; dashed line marks linear_ridge | GSE132080 | seed 7 | no uncertainty bars (single seed)", fontsize=9)
    legend_handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in [("Reference", GROUP_COLORS["reference"]), ("Linear", GROUP_COLORS["linear"]), ("Deep learning", GROUP_COLORS["deep"]), ("Diff-AE", GROUP_COLORS["diffae"]), ("Diff-AE repair", GROUP_COLORS["repair"])]]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.015))
    save_figure(fig, output_dir, "gse132080_vs_linear_ridge")


def plot_win_counts(wins: pd.DataFrame, output_dir: Path) -> None:
    win_counts = wins.set_index("method")["wins"].reindex(METHOD_ORDER, fill_value=0)
    win_counts = win_counts[win_counts.gt(0)].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [GROUP_COLORS[METHOD_GROUPS[method]] for method in win_counts.index]
    bars = ax.barh([METHOD_LABELS[m] for m in win_counts.index], win_counts.values, color=colors)
    for bar, value in zip(bars, win_counts.values):
        ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=10)
    ax.set_xlim(0, max(4, int(win_counts.max()) + 1))
    ax.set_xlabel("Number of rank-1 metric/budget cells")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.suptitle("Top-1 wins across the 10 reported metric/budget cells", x=0.01, ha="left", y=0.98, weight="bold")
    fig.text(0.01, 0.91, "Counts recomputed from rankings.csv and checked against wins.csv", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    save_figure(fig, output_dir, "gse132080_top1_wins")


def write_manifest(output_dir: Path, aggregate_dir: Path) -> None:
    manifest = {
        "dataset": "GSE132080",
        "split": "unseen-guide/seen-target",
        "held_out_conditions": 15,
        "genes": 1019,
        "seed": 7,
        "source": str(aggregate_dir),
        "excluded": ["all squidiff variants", "fix_eps_* sampling failures"],
        "figures": [
            "gse132080_rank_heatmap",
            "gse132080_vs_linear_ridge",
            "gse132080_top1_wins",
        ],
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary, rankings, wins = load_data(args.aggregate_dir)
    displayed = validate_data(summary, rankings, wins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_rank_heatmap(rankings, args.output_dir)
    plot_baseline_comparison(displayed, args.output_dir)
    plot_win_counts(wins, args.output_dir)
    write_manifest(args.output_dir, args.aggregate_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
