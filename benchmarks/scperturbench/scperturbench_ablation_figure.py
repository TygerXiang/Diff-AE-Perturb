"""Paired statistics and publication-quality figures for the GSE132080 activity ablation.

Compares two Diff-AE-Perturb checkpoints that differ only in ``--activity-source``
(``predicted`` sgRNA-efficiency conditioning vs none) on the identical target-level
unseen-guide benchmark (15 conditions, 1019 genes, DDIM 50, seed 7).

Outputs (under ``--output-dir``):

    ablation_pairwise.csv        per-metric arm means/medians, paired deltas, bootstrap CIs
    ablation_tests.csv           Wilcoxon signed-rank + paired t-test, effect sizes, BH q-values
    ablation_per_condition.csv   long-form per-condition values for both arms
    ablation_summary_wide.csv    2 arms x 6 metrics, ready for a manuscript table
    figures/fig1_activity_ablation_6metrics.{png,pdf}
    figures/fig2_activity_ablation_6metrics_livechannel.{png,pdf}
    figures/fig3_per_condition_deltas.{png,pdf}

Statistical design
------------------
* Unit of analysis is the *condition* (n=15), not the cell: cells within a condition share the
  same control pool and the same target, so cell-level resampling would overstate precision.
* Bars report the mean with a 95% bootstrap CI over conditions (BCa-free percentile interval).
* Arm comparison uses the paired Wilcoxon signed-rank test (non-parametric, the metrics are
  skewed) with Benjamini-Hochberg correction across the six metrics, plus a paired t-test and
  Hedges' g on the paired differences for readers who prefer parametric summaries.
* ``fig2`` repeats everything on the 12 conditions whose held-out guides actually carry Table S8
  CNN features, i.e. where the activity channel is live (see the preflight report).  The three
  reference-only conditions (GNB2L1, HSPA9, POLR1D) receive zero activity features by
  construction, so they can only dilute the effect; both views are reported rather than picking one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------------------
# metric definition: per-condition pairing key -> (metric, deg budget)
# ---------------------------------------------------------------------------

PRIMARY_METRICS = [
    ("mse", "100", "MSE", False),
    ("pcc_delta", "100", "PCC of delta", True),
    ("e_distance", "100", "Energy distance", False),
    ("wasserstein", "100", "Wasserstein", False),
    # The literal sym-KL (``kl_divergence``) is NOT usable here: on this data its
    # ``(x_mean - y_mean)^2 / (2 y_std^2)`` term is dominated by single genes whose stimulated
    # standard deviation is exactly zero (epsilon = 1e-8).  For this benchmark that one gene
    # supplied 77.6% of the metric on the "high" conditions -- RPS14/POLR2H/HSPE1 reached 100% --
    # which splits the 15 conditions into an artifact-driven bimodal pattern (see
    # reports/assets/activity_ablation/kl_bimodality_diagnosis.txt).  The std-floored companion
    # (``kl_floor``, std floor 0.1) is the same statistic with that degenerate term bounded, so it
    # is what this figure reports.
    ("kl_floor", "100", "Symmetric KL", False),
    ("common_degs", "cutoff", "Common DEGs", True),
]
SECONDARY_METRICS = [
    ("mse", "all", "MSE", False),
    ("pcc_delta", "all", "PCC of delta", True),
    ("e_distance", "all", "Energy distance", False),
]

ARM_ON = "activity_on_200ep"
ARM_OFF = "activity_off_200ep"
COLOR_ON = "#2F6FB5"      # colour-blind safe blue
COLOR_OFF = "#D1793A"     # colour-blind safe orange
COLOR_LIVE = "#3C8C5A"
COLOR_DEAD = "#9AA0A6"

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260930


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse and plot the GSE132080 activity ablation.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent
        / ".scperturbench/results/gse132080_activity_ablation/gse132080",
    )
    parser.add_argument("--dataset", default="gse132080")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arm-on", default=ARM_ON)
    parser.add_argument("--arm-off", default=ARM_OFF)
    parser.add_argument("--preflight", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="GSE132080 · predicted sgRNA efficiency conditioning")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, b: int = BOOTSTRAP_B,
                 alpha: float = 0.05) -> tuple[float, float]:
    if values.size < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, values.size, size=(b, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_paired_delta_ci(on: np.ndarray, off: np.ndarray, rng: np.random.Generator,
                              b: int = BOOTSTRAP_B, alpha: float = 0.05) -> tuple[float, float]:
    if on.size < 2:
        return (float("nan"), float("nan"))
    diff = on - off
    idx = rng.integers(0, diff.size, size=(b, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def hedges_g_paired(diff: np.ndarray) -> float:
    n = diff.size
    if n < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("nan")
    d = diff.mean() / sd
    correction = 1 - 3 / (4 * n - 5)
    return float(d * correction)


def rank_biserial_paired(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    total = ranks.sum()
    return float((r_plus - r_minus) / total) if total else float("nan")


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        q[order[i]] = running
    return [float(min(max(value, 0.0), 1.0)) for value in q]


def load_arm(results_root: Path, arm: str, seed: int) -> pd.DataFrame:
    path = results_root / arm / f"seed{seed}" / "score" / "per_condition.csv"
    if not path.exists():
        raise SystemExit(f"missing per-condition scores: {path}")
    frame = pd.read_csv(path)
    frame["arm"] = arm
    return frame


def pivot_metric(frame: pd.DataFrame, metric: str, budget: str) -> pd.DataFrame:
    if metric == "kl_floor":
        # The std-floored companion is stored on the kl_divergence rows in the ``kl_floor`` column
        # (already log2(x+1)-transformed by the scorer), not as its own metric row.
        subset = frame[(frame["metric"] == "kl_divergence") & (frame["deg_budget"].astype(str) == budget)]
        subset = subset.dropna(subset=["kl_floor"])
        wide = subset.pivot_table(index="condition", columns="arm", values="kl_floor", aggfunc="mean")
        return wide.dropna(how="any")
    subset = frame[(frame["metric"] == metric) & (frame["deg_budget"].astype(str) == budget)]
    wide = subset.pivot_table(index="condition", columns="arm", values="value", aggfunc="mean")
    return wide.dropna(how="any")


def build_analysis(frame: pd.DataFrame, arm_on: str, arm_off: str, rng: np.random.Generator,
                   metrics: list[tuple[str, str, str, bool]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []

    for metric, budget, label, higher_better in metrics:
        wide = pivot_metric(frame, metric, budget)
        if wide.empty:
            continue
        on = wide[arm_on].to_numpy(dtype=float)
        off = wide[arm_off].to_numpy(dtype=float)
        diff = on - off
        ci_on = bootstrap_ci(on, rng)
        ci_off = bootstrap_ci(off, rng)
        ci_diff = bootstrap_paired_delta_ci(on, off, rng)

        try:
            wilcoxon = stats.wilcoxon(on, off, zero_method="wilcox", alternative="two-sided")
            p_wilcoxon = float(wilcoxon.pvalue)
            n_zero = int((diff == 0).sum())
            statistic = float(wilcoxon.statistic)
        except ValueError:
            p_wilcoxon = float("nan")
            statistic = float("nan")
            n_zero = int((diff == 0).sum())

        ttest = stats.ttest_rel(on, off)
        pair_rows.append(
            {
                "metric": metric,
                "deg_budget": budget,
                "label": label,
                "higher_is_better": higher_better,
                "n_conditions": int(on.size),
                "mean_activity_on": float(on.mean()),
                "mean_activity_off": float(off.mean()),
                "median_activity_on": float(np.median(on)),
                "median_activity_off": float(np.median(off)),
                "bootstrap_ci_low_activity_on": ci_on[0],
                "bootstrap_ci_high_activity_on": ci_on[1],
                "bootstrap_ci_low_activity_off": ci_off[0],
                "bootstrap_ci_high_activity_off": ci_off[1],
                "mean_paired_delta_on_minus_off": float(diff.mean()),
                "paired_delta_ci_low": ci_diff[0],
                "paired_delta_ci_high": ci_diff[1],
                "relative_change_pct": float(100 * diff.mean() / off.mean()) if off.mean() != 0 else float("nan"),
                "n_conditions_activity_on_better": int(
                    (diff > 0).sum() if higher_better else (diff < 0).sum()
                ),
                "n_conditions_tied": n_zero,
            }
        )
        test_rows.append(
            {
                "metric": metric,
                "deg_budget": budget,
                "n_conditions": int(on.size),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_wilcoxon,
                "paired_t_statistic": float(ttest.statistic),
                "paired_t_p": float(ttest.pvalue),
                "hedges_g_paired": hedges_g_paired(diff),
                "rank_biserial_paired": rank_biserial_paired(diff),
            }
        )
        for condition, row in wide.iterrows():
            long_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "deg_budget": budget,
                    "value_activity_on": float(row[arm_on]),
                    "value_activity_off": float(row[arm_off]),
                    "delta_on_minus_off": float(row[arm_on] - row[arm_off]),
                    "higher_is_better": higher_better,
                }
            )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["wilcoxon_q_bh"] = benjamini_hochberg(tests["wilcoxon_p"].tolist())
        tests["paired_t_q_bh"] = benjamini_hochberg(tests["paired_t_p"].tolist())
        tests["significant_q_lt_0.05"] = tests["wilcoxon_q_bh"] < 0.05

    pairwise = pd.DataFrame(pair_rows)
    if not pairwise.empty and not tests.empty:
        pairwise = pairwise.merge(
            tests[
                [
                    "metric",
                    "deg_budget",
                    "wilcoxon_p",
                    "wilcoxon_q_bh",
                    "paired_t_p",
                    "hedges_g_paired",
                    "rank_biserial_paired",
                    "significant_q_lt_0.05",
                ]
            ],
            on=["metric", "deg_budget"],
            how="left",
        )
    return pairwise, tests, pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def significance_text(q: float, p: float) -> str:
    if not np.isfinite(q):
        return "n/a"
    if q < 0.001:
        return "q<0.001"
    if q < 0.01:
        return f"q={q:.3f}"
    if q < 0.05:
        return f"q={q:.3f}"
    return f"q={q:.2f} (n.s.)"


def panel_axes_layout(n: int, *, columns: int = 3, width: float = 12.2, panel_height: float = 3.5):
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(width, panel_height * rows))
    axes = np.atleast_1d(axes).ravel()
    return fig, axes


def plot_metrics(
    long_frame: pd.DataFrame,
    pairwise: pd.DataFrame,
    *,
    metrics: list[tuple[str, str, str, bool]],
    arm_on: str,
    arm_off: str,
    title: str,
    out_base: Path,
    dpi: int,
    subtitle: str,
    arm_on_label: str = "predicted sgRNA efficiency",
    arm_off_label: str = "predicted sgRNA efficiency",
    minimal: bool = False,
) -> None:
    """Six-metric paired bar figure.

    ``minimal=True`` strips every piece of prose from the panels - no subtitle, no
    parenthetical budget note under a metric name, no statistics block - and uses
    one-line category labels (``with activity`` / ``without activity``).  The bars,
    bootstrap error bars and paired per-guide overlay are unchanged; only text is
    removed, so the figure stays the same shape as the GSE132080 version.
    """

    def panel_title(label: str) -> str:
        if not minimal:
            return label
        # Drop the trailing "(...)" so the panel reads "MSE" not "MSE (DEG-100)".
        return label.split(" (")[0].strip()

    n = len(metrics)
    fig, axes = panel_axes_layout(n)

    for ax, (metric, budget, label, higher_better) in zip(axes, metrics):
        pair_row = pairwise[(pairwise["metric"] == metric) & (pairwise["deg_budget"] == budget)]
        subset = long_frame[(long_frame["metric"] == metric) & (long_frame["deg_budget"] == budget)]
        if pair_row.empty or subset.empty:
            ax.set_visible(False)
            continue
        pair_row = pair_row.iloc[0]

        on = subset["value_activity_on"].to_numpy(dtype=float)
        off = subset["value_activity_off"].to_numpy(dtype=float)
        conditions = subset["condition"].tolist()

        # Left bar = "with", right bar = "without": the same order as the legend, so the figure never
        # reads backwards (bars are laid out on x, not on a categorical axis).
        x_on, x_off = 0.0, 1.0
        width = 0.52
        means = [on.mean(), off.mean()]
        errs = np.array(
            [
                [on.mean() - pair_row["bootstrap_ci_low_activity_on"],
                 off.mean() - pair_row["bootstrap_ci_low_activity_off"]],
                [pair_row["bootstrap_ci_high_activity_on"] - on.mean(),
                 pair_row["bootstrap_ci_high_activity_off"] - off.mean()],
            ]
        )
        errs = np.clip(errs, 0, None)

        ax.bar(
            [x_on, x_off],
            means,
            width=width,
            color=[COLOR_ON, COLOR_OFF],
            edgecolor="black",
            linewidth=0.9,
            zorder=2,
        )
        ax.errorbar(
            [x_on, x_off],
            means,
            yerr=errs,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=4,
            zorder=4,
        )

        # paired per-condition overlay: every condition appears in both bars, joined by a thin line
        jitter_rng = np.random.default_rng(11)
        for idx, condition in enumerate(conditions):
            jitter = jitter_rng.uniform(-0.075, 0.075)
            ax.plot(
                [x_on + jitter, x_off + jitter],
                [on[idx], off[idx]],
                color="#5B5B5B",
                linewidth=0.5,
                alpha=0.55,
                zorder=3,
            )
            ax.scatter(
                [x_on + jitter, x_off + jitter],
                [on[idx], off[idx]],
                s=11,
                facecolor="white",
                edgecolor="#3A3A3A",
                linewidth=0.6,
                zorder=5,
            )

        # Title carries the metric name only.  The better-direction note belongs in the caption and the
        # per-metric test results live in ablation_tests.csv, so the panel itself stays uncluttered.
        ax.set_title(panel_title(label), fontsize=10.0, pad=6)
        ax.set_xticks([x_on, x_off])
        # The GSE283614 ablation conditions on a *measured* screen value, so the
        # labels are parameterised rather than hardcoded to "predicted".
        if minimal:
            ax.set_xticklabels(["with activity", "without activity"], fontsize=8.5)
        else:
            ax.set_xticklabels([f"with\n{arm_on_label}", f"without\n{arm_off_label}"], fontsize=8)
        style_axes(ax)
        ax.set_xlim(-0.45, 1.45)
        if minimal:
            # Bar panels should read from a zero baseline.
            ax.set_ylim(bottom=0.0)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], color=COLOR_ON, lw=7, label="with activity"),
        Line2D([0], [0], color=COLOR_OFF, lw=7, label="without activity"),
        Line2D([0], [0], color="#5B5B5B", lw=0.8, marker="o", markerfacecolor="white",
               markeredgecolor="#3A3A3A", markersize=4.5, label="individual guide (paired)"),
    ] if minimal else [
        Line2D([0], [0], color=COLOR_ON, lw=7, label=f"with {arm_on_label} ({arm_on})"),
        Line2D([0], [0], color=COLOR_OFF, lw=7, label=f"without {arm_off_label} ({arm_off})"),
        Line2D([0], [0], color="#5B5B5B", lw=0.8, marker="o", markerfacecolor="white",
               markeredgecolor="#3A3A3A", markersize=4.5, label="individual held-out condition (paired)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=11.5, y=0.995)
    if subtitle and not minimal:
        fig.text(0.5, 0.958, subtitle, ha="center", fontsize=7.8, color="#444444")
    # Without a subtitle there is no text row to reserve space for.
    top = 0.95 if (subtitle and not minimal) else 0.985
    fig.tight_layout(rect=(0, 0.045, 1, top))
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base.with_suffix('.png')} and .pdf")


def plot_per_condition_deltas(
    long_frame: pd.DataFrame,
    *,
    metrics: list[tuple[str, str, str, bool]],
    arm_on: str,
    arm_off: str,
    title: str,
    out_base: Path,
    dpi: int,
    live_conditions: list[str] | None,
) -> None:
    """Per-condition relative change, so the reader sees the distribution behind each bar."""
    n = len(metrics)
    fig, axes = panel_axes_layout(n, columns=3, width=12.2, panel_height=3.3)
    for ax, (metric, budget, label, higher_better) in zip(axes, metrics):
        subset = long_frame[(long_frame["metric"] == metric) & (long_frame["deg_budget"] == budget)].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        subset["relative_pct"] = 100 * (
            subset["value_activity_on"] - subset["value_activity_off"]
        ) / subset["value_activity_off"].replace(0, np.nan)
        subset = subset.sort_values("relative_pct")
        colours = [
            COLOR_ON if (value > 0) == higher_better else COLOR_OFF
            for value in subset["relative_pct"]
        ]
        ax.barh(
            np.arange(subset.shape[0]),
            subset["relative_pct"],
            color=colours,
            edgecolor="black",
            linewidth=0.6,
            height=0.72,
        )
        labels = [
            condition + ("" if live_conditions is None or condition in live_conditions else " *")
            for condition in subset["condition"]
        ]
        ax.set_yticks(np.arange(subset.shape[0]))
        ax.set_yticklabels(labels, fontsize=6.4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{label}", fontsize=9)
        ax.set_xlabel("relative change with activity (%)", fontsize=7.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], color=COLOR_ON, lw=7, label="moves toward better"),
        Line2D([0], [0], color=COLOR_OFF, lw=7, label="moves toward worse"),
    ]
    note = "sorted by relative change; * = condition whose held-out guide has no Table S8 CNN features (activity channel inert)"
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=11.5, y=0.995)
    fig.text(0.5, 0.955, note, ha="center", fontsize=7.2, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 0.945))
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base.with_suffix('.png')} and .pdf")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    results_root = args.results_root
    output_dir = args.output_dir or results_root / "ablation_analysis"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    frame = pd.concat(
        [
            load_arm(results_root, args.arm_off, args.seed),
            load_arm(results_root, args.arm_on, args.seed),
        ],
        ignore_index=True,
    )

    live_conditions: list[str] | None = None
    feature_less: list[str] = []
    if args.preflight is not None and Path(args.preflight).exists():
        preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
        activity = preflight.get("activity", {})
        live_conditions = activity.get("conditions_with_feature_bearing_guides")
        feature_less = activity.get("conditions_without_feature_bearing_guides", [])

    all_metrics = PRIMARY_METRICS + SECONDARY_METRICS
    pairwise, tests, long_frame = build_analysis(frame, args.arm_on, args.arm_off, rng, all_metrics)
    tests = tests.merge(
        pairwise[["metric", "deg_budget", "label", "higher_is_better", "mean_activity_on",
                  "mean_activity_off", "mean_paired_delta_on_minus_off", "relative_change_pct"]],
        on=["metric", "deg_budget"],
        how="left",
    )

    pairwise.to_csv(output_dir / "ablation_pairwise.csv", index=False)
    tests.to_csv(output_dir / "ablation_tests.csv", index=False)
    long_frame.to_csv(output_dir / "ablation_per_condition.csv", index=False)

    wide = long_frame.copy()
    wide["metric_budget"] = wide["metric"] + "@" + wide["deg_budget"]
    wide_summary = (
        wide.groupby(["metric_budget", "higher_is_better"])[["value_activity_on", "value_activity_off"]]
        .mean()
        .reset_index()
        .rename(columns={"value_activity_on": f"mean_{args.arm_on}", "value_activity_off": f"mean_{args.arm_off}"})
    )
    wide_summary["mean_paired_delta_on_minus_off"] = (
        wide_summary[f"mean_{args.arm_on}"] - wide_summary[f"mean_{args.arm_off}"]
    )
    wide_summary.to_csv(output_dir / "ablation_summary_wide.csv", index=False)

    # ---- figures ---------------------------------------------------------------------------
    # minimal=True: one-line "with activity" / "without activity" category labels, metric name only
    # as the panel title, no subtitle and no statistics block (those live in ablation_tests.csv).
    plot_metrics(
        long_frame,
        pairwise,
        metrics=PRIMARY_METRICS,
        arm_on=args.arm_on,
        arm_off=args.arm_off,
        title="GSE132080",
        out_base=figures_dir / "fig1_activity_ablation_6metrics",
        dpi=args.dpi,
        subtitle="",
        minimal=True,
    )
    plot_per_condition_deltas(
        long_frame,
        metrics=PRIMARY_METRICS,
        arm_on=args.arm_on,
        arm_off=args.arm_off,
        title=f"{args.title} · per-condition effect",
        out_base=figures_dir / "fig3_per_condition_deltas",
        dpi=args.dpi,
        live_conditions=live_conditions,
    )

    if live_conditions:
        live_long = long_frame[long_frame["condition"].isin(live_conditions)].copy()
        live_rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
        live_pairwise, live_tests, _ = build_analysis(
            frame[frame["condition"].isin(live_conditions)], args.arm_on, args.arm_off, live_rng, PRIMARY_METRICS
        )
        live_pairwise.to_csv(output_dir / "ablation_pairwise_livechannel.csv", index=False)
        live_tests.to_csv(output_dir / "ablation_tests_livechannel.csv", index=False)
        plot_metrics(
            live_long,
            live_pairwise,
            metrics=PRIMARY_METRICS,
            arm_on=args.arm_on,
            arm_off=args.arm_off,
            title="GSE132080 · activity channel live",
            out_base=figures_dir / "fig2_activity_ablation_6metrics_livechannel",
            dpi=args.dpi,
            subtitle=(
                f"restricted to the {len(live_conditions)} conditions whose held-out guides carry Table S8 CNN "
                f"features; excluded (channel inert): {', '.join(feature_less) if feature_less else 'none'}"
            ),
        )

    # ---- console summary -------------------------------------------------------------------
    print("\n=== primary metrics (all 15 conditions) ===")
    view = tests[tests["deg_budget"].isin(["100", "cutoff"])].copy()
    view = view[view["metric"].isin([m for m, _, _, _ in PRIMARY_METRICS])]
    cols = ["metric", "mean_activity_off", "mean_activity_on", "mean_paired_delta_on_minus_off",
            "relative_change_pct", "wilcoxon_p", "wilcoxon_q_bh", "hedges_g_paired"]
    print(view[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nanalysis artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
