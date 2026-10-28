"""Aggregate per-method metric summaries into the benchmark table.

Reads every ``<results-root>/<dataset>/<method>/seed<N>/score/summary.csv`` and writes

    aggregate/summary_long.csv    method, seed, metric, budget, value
    aggregate/summary_wide.csv    one row per method, one column per metric@budget
    aggregate/rankings.csv        per-metric rank and win counts
    aggregate/report.md           human-readable markdown table

Runs anywhere (numpy/pandas only); no container required beyond reading the files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["mse", "pcc_delta", "e_distance", "wasserstein", "kl_divergence", "common_degs"]
HIGHER_IS_BETTER = {
    "mse": False,
    "pcc_delta": True,
    "e_distance": False,
    "wasserstein": False,
    "kl_divergence": False,
    "common_degs": True,
}


def load_summaries(results_root: Path, dataset: str) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(results_root.glob(f"{dataset}/*/seed*/score/summary.csv")):
        frame = pd.read_csv(summary_path)
        for record in frame.to_dict("records"):
            metric_budget = str(record["metric_budget"])
            metric, _, budget = metric_budget.partition("@")
            rows.append(
                {
                    "dataset": dataset,
                    "method": record["method"],
                    "seed": int(record["seed"]),
                    "metric": metric,
                    "budget": budget,
                    "value": float(record["value"]),
                    "source": str(summary_path),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate six-metric benchmark summaries.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", default="gse132080")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="*", default=METRICS)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="methods to keep in the long/wide CSVs but drop from the ranked report",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_summaries(args.results_root, args.dataset)
    if frame.empty:
        raise SystemExit(f"no summaries found under {args.results_root}/{args.dataset}")
    frame["metric_budget"] = frame["metric"] + "@" + frame["budget"]
    frame.to_csv(args.output_dir / "summary_long.csv", index=False)

    all_methods = sorted(frame["method"].unique())
    if args.exclude:
        frame = frame[~frame["method"].isin(args.exclude)]
        (args.output_dir / "excluded_methods.txt").write_text(
            "\n".join(args.exclude), encoding="utf-8"
        )

    mean = (
        frame.groupby(["method", "metric_budget"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    mean.to_csv(args.output_dir / "summary_stats.csv", index=False)

    wide = mean.pivot(index="method", columns="metric_budget", values="mean")
    wide.to_csv(args.output_dir / "summary_wide.csv")

    # per-metric ranking over methods (mean over seeds first)
    rank_rows = []
    for metric_budget, group in mean.groupby("metric_budget"):
        metric = metric_budget.split("@")[0]
        if metric not in HIGHER_IS_BETTER:
            continue
        ordered = group.sort_values("mean", ascending=not HIGHER_IS_BETTER[metric])
        for rank, (_, record) in enumerate(ordered.iterrows(), start=1):
            rank_rows.append(
                {
                    "metric_budget": metric_budget,
                    "rank": rank,
                    "method": record["method"],
                    "mean": record["mean"],
                    "std": record["std"],
                    "n_seeds": int(record["count"]),
                }
            )
    rankings = pd.DataFrame(rank_rows)
    rankings.to_csv(args.output_dir / "rankings.csv", index=False)

    if not rankings.empty:
        avg_rank = (
            rankings[rankings["rank"] == 1]
            .groupby("method")
            .size()
            .rename("wins")
            .reset_index()
        )
        avg_rank.to_csv(args.output_dir / "wins.csv", index=False)

    lines = [f"# {args.dataset} six-metric benchmark", ""]
    lines.append(f"- methods: {', '.join(sorted(frame['method'].unique()))}")
    if args.exclude and len(all_methods) != len(frame['method'].unique()):
        lines.append(f"- excluded from ranking: {', '.join(args.exclude)}")
    lines.append(f"- seeds: {', '.join(str(s) for s in sorted(frame['seed'].unique()))}")
    lines.append(f"- metrics: {', '.join(args.metrics)}")
    lines.append("")
    lines.append("## Mean over evaluated conditions")
    lines.append("")
    for metric_budget in sorted(mean["metric_budget"].unique()):
        metric = metric_budget.split("@")[0]
        if metric not in args.metrics:
            continue
        subset = mean[mean["metric_budget"] == metric_budget].sort_values("mean", ascending=not HIGHER_IS_BETTER.get(metric, False))
        lines.append(f"### {metric_budget} ({'higher' if HIGHER_IS_BETTER.get(metric) else 'lower'} is better)")
        lines.append("")
        lines.append("| rank | method | mean | sd | n |")
        lines.append("|---|---|---|---|---|")
        for rank, (_, record) in enumerate(subset.iterrows(), start=1):
            sd = record["std"]
            sd_text = "" if pd.isna(sd) else f"{sd:.6f}"
            lines.append(f"| {rank} | {record['method']} | {record['mean']:.6f} | {sd_text} | {int(record['count'])} |")
        lines.append("")
    if not rankings.empty:
        lines.append("## Top-1 counts across metric/budget cells")
        lines.append("")
        wins = rankings[rankings["rank"] == 1].groupby("method").size().sort_values(ascending=False)
        lines.append("| method | wins |")
        lines.append("|---|---|")
        for method, count in wins.items():
            lines.append(f"| {method} | {int(count)} |")
        lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
