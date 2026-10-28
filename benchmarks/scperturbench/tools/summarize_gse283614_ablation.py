"""Collect every GSE283614 ablation result into one summary table.

Reads the analysis outputs written by ``scperturbench_gse283614_ablation.py`` and
the probe CSVs, then emits:

    summary_all.csv          one row per (state, comparison, metric, budget)
    summary_all.md           markdown tables for the report
    probe_summary.csv        noise-controlled activity-perturbation probe roll-up
    seed_variance.csv        between-seed vs between-guide spread (A4)

Nothing here recomputes statistics; it only re-presents what the per-stage
drivers produced, so the report and the CSVs cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMPARISONS = {
    "esc": [
        ("activity_on vs activity_off", "esc/ablation_analysis", "activity_on", "activity_off"),
        ("activity_perm vs activity_off", "esc/ablation_analysis_permutation", "activity_perm", "activity_off"),
        ("activity_mask vs activity_off", "esc/ablation_analysis_mask", "activity_mask", "activity_off"),
    ],
    "de": [
        ("activity_on vs activity_off", "de/ablation_analysis", "activity_on", "activity_off"),
        # The DE mask arm exists because it is the control that separates "the
        # values are uninformative" from "the measurement pattern itself is
        # misleading" - and on DE both arms degrade, which is the point.
        ("activity_mask vs activity_off", "de/ablation_analysis_mask", "activity_mask", "activity_off"),
    ],
}

# Which arm each analysis directory compares against the baseline.  Used to
# discover comparisons that exist on disk but are not in COMPARISONS (e.g. the
# knockdown-stratum run, which has its own result tree and arm set).
ANALYSIS_ARMS = {
    "ablation_analysis": "activity_on",
    "ablation_analysis_mask": "activity_mask",
    "ablation_analysis_permutation": "activity_perm",
}

PRIMARY = ["mse@100", "pcc_delta@100", "e_distance@100", "wasserstein@100", "kl_divergence@100", "common_degs@cutoff"]
SECONDARY = ["mse@all", "pcc_delta@all", "e_distance@all", "kl_divergence@all"]

HIGHER_IS_BETTER = {
    "mse": False,
    "pcc_delta": True,
    "e_distance": False,
    "wasserstein": False,
    "kl_divergence": False,
    "common_degs": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise the GSE283614 ablation results.")
    parser.add_argument(
        "--results-root", type=Path,
        default=Path("/root/perturbench/.scperturbench/results/gse283614_activity_ablation/gse283614"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, default=None)
    parser.add_argument("--diagnostic", type=Path, default=None)
    parser.add_argument("--discover", action="store_true",
                        help="Also collect any ablation_analysis* directory found under each state, "
                             "instead of only the hard-coded comparison list.")
    return parser.parse_args()


def discover_comparisons(results_root: Path) -> dict[str, list[tuple[str, str, str, str]]]:
    """Find every analysis directory per state and pair it with the baseline arm.

    The directory name encodes the arm under test (``ablation_analysis_mask`` ->
    ``activity_mask``), so a run that trains a different arm set needs no code
    change here - it just shows up.
    """

    found: dict[str, list[tuple[str, str, str, str]]] = {}
    for state_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        state = state_dir.name
        entries: list[tuple[str, str, str, str]] = []
        for analysis in sorted(state_dir.glob("ablation_analysis*")):
            if not (analysis / "ablation_tests.csv").exists():
                continue
            suffix = analysis.name.replace("ablation_analysis", "").lstrip("_")
            if not suffix:
                arm = "activity_on"
            else:
                arm = ANALYSIS_ARMS.get(analysis.name, f"activity_{suffix}")
            label = f"{arm} vs activity_off"
            entries.append((label, f"{state}/{analysis.name}", arm, "activity_off"))
        if entries:
            found[state] = entries
    return found


def collect(results_root: Path, *, discover: bool = False) -> pd.DataFrame:
    rows = []
    comparisons = discover_comparisons(results_root) if discover else COMPARISONS
    for state, state_comparisons in comparisons.items():
        for label, rel, arm_on, arm_off in state_comparisons:
            analysis = results_root / rel
            tests_path = analysis / "ablation_tests.csv"
            if not tests_path.exists():
                print(f"  missing {tests_path}; skipping {state}/{label}")
                continue
            tests = pd.read_csv(tests_path)
            for _, row in tests.iterrows():
                budget = f"{row['metric']}@{row['deg_budget']}"
                direction = +1 if HIGHER_IS_BETTER.get(row["metric"], False) else -1
                delta_on_minus_off = float(row["mean_paired_delta_on_minus_off"])
                rows.append(
                    {
                        "state": state,
                        "comparison": label,
                        "arm_on": arm_on,
                        "arm_off": arm_off,
                        "metric": row["metric"],
                        "deg_budget": row["deg_budget"],
                        "metric_budget": budget,
                        "n_conditions": int(row["n_conditions"]),
                        "mean_arm_on": float(row["mean_activity_on"]),
                        "mean_arm_off": float(row["mean_activity_off"]),
                        "mean_paired_delta": delta_on_minus_off,
                        "relative_change_pct": float(row["relative_change_pct"]),
                        # "better" is defined by the metric's direction, never by the sign of the delta.
                        "arm_on_better": bool(direction * delta_on_minus_off > 0),
                        "wilcoxon_p": float(row["wilcoxon_p"]),
                        "wilcoxon_q_bh": float(row["wilcoxon_q_bh"]),
                        "paired_t_p": float(row["paired_t_p"]),
                        "hedges_g_paired": float(row["hedges_g_paired"]),
                        "significant_q_lt_0.05": bool(row.get("significant_q_lt_0.05", row["wilcoxon_q_bh"] < 0.05)),
                    }
                )
    return pd.DataFrame(rows)


def collect_subsets(results_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_root.glob("*/ablation_analysis*/ablation_subsets.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "state", path.relative_to(results_root).parts[0])
        frame.insert(1, "analysis", path.parent.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_probe(results_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_root.glob("*/fold_*/activity_on/activity_perturbation_probe_seed*.csv")):
        frame = pd.read_csv(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    probe = pd.concat(frames, ignore_index=True)
    # A swap between two identical values must produce a zero shift; if it does
    # not, the probe is measuring sampling noise rather than the channel.
    probe["zero_delta_swap"] = probe["activity_delta"].abs() < 1e-12
    return probe


def probe_summary(probe: pd.DataFrame) -> pd.DataFrame:
    if probe.empty:
        return probe
    rows = []
    for state, group in probe.groupby("state"):
        zero = group[group["zero_delta_swap"]]
        moved = group[~group["zero_delta_swap"]]
        rows.append(
            {
                "state": state,
                "n_swaps": int(len(group)),
                "n_zero_delta_swaps": int(len(zero)),
                "max_shift_on_zero_delta_swap": float(zero["l2_shift_swapped"].max()) if len(zero) else np.nan,
                "median_shift": float(moved["l2_shift_swapped"].median()) if len(moved) else np.nan,
                "median_between_guide": float(group["l2_between_guide"].median()),
                "median_noise_floor": float(group["l2_noise_floor"].median()),
                "median_shift_over_between_guide": float(moved["ratio_shift_over_between_guide"].median()) if len(moved) else np.nan,
                "median_shift_over_noise_floor": float(moved["ratio_shift_over_noise_floor"].median()) if len(moved) else np.nan,
                "n_swaps_above_noise_floor": int((moved["ratio_shift_over_noise_floor"] > 1).sum()) if len(moved) else 0,
            }
        )
    return pd.DataFrame(rows)


def collect_seed_variance(results_root: Path, state: str = "esc") -> pd.DataFrame:
    """Between-seed spread vs between-guide spread for the activity arm."""

    state_dir = results_root / state
    if not state_dir.exists():
        return pd.DataFrame()
    # seed 7 lives under activity_on/, extra seeds under activity_on_seed<k>/.
    arm_dir_by_seed: dict[int, Path] = {}
    for path in sorted(state_dir.glob("fold_0/activity_on*")):
        if path.name == "activity_on":
            arm_dir_by_seed[7] = path
            continue
        try:
            seed = int(path.name.split("seed")[1])
        except (IndexError, ValueError):
            continue
        arm_dir_by_seed[seed] = path
    if len(arm_dir_by_seed) < 2:
        return pd.DataFrame()

    per_seed: dict[int, pd.Series] = {}
    for seed, arm_dir in arm_dir_by_seed.items():
        frames = []
        for score in sorted(state_dir.glob(f"fold_*/{arm_dir.name}/seed{seed}/score/per_condition.csv")):
            frames.append(pd.read_csv(score))
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        subset = frame[(frame["metric"] == "mse") & (frame["deg_budget"].astype(str) == "100")]
        per_seed[seed] = subset.set_index("condition")["value"]
    if len(per_seed) < 2:
        return pd.DataFrame()

    shared = set.intersection(*(set(series.index) for series in per_seed.values()))
    if len(shared) < 3:
        return pd.DataFrame()
    matrix = pd.DataFrame({seed: series.loc[sorted(shared)] for seed, series in per_seed.items()})
    between_seed = matrix.std(axis=1, ddof=1).median()
    between_guide = matrix.mean(axis=1).std(ddof=1)
    return pd.DataFrame(
        [
            {
                "state": state,
                "metric": "mse@100",
                "n_seeds": int(matrix.shape[1]),
                "n_guides": int(matrix.shape[0]),
                "median_within_guide_between_seed_sd": float(between_seed),
                "between_guide_sd_of_seed_means": float(between_guide),
                "ratio_seed_over_guide": float(between_seed / between_guide) if between_guide else np.nan,
            }
        ]
    )


def markdown_tables(frame: pd.DataFrame) -> str:
    lines: list[str] = []
    for (state, comparison), group in frame.groupby(["state", "comparison"], sort=False):
        lines.append(f"### {state.upper()} - {comparison}")
        lines.append("")
        lines.append("| metric | n | arm_on | arm_off | delta | rel % | better? | Wilcoxon p | BH q | t p | Hedges g |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        ordered = [m for m in PRIMARY + SECONDARY if m in set(group["metric_budget"])]
        for metric_budget in ordered:
            row = group[group["metric_budget"] == metric_budget]
            if row.empty:
                continue
            row = row.iloc[0]
            lines.append(
                "| {mb} | {n} | {on:.5g} | {off:.5g} | {d:+.5g} | {rel:+.1f}% | {better} | {wp:.4g} | {q:.4g} | {tp:.4g} | {g:+.3f} |".format(
                    mb=metric_budget,
                    n=int(row["n_conditions"]),
                    on=row["mean_arm_on"],
                    off=row["mean_arm_off"],
                    d=row["mean_paired_delta"],
                    rel=row["relative_change_pct"],
                    better="yes" if row["arm_on_better"] else "no",
                    wp=row["wilcoxon_p"],
                    q=row["wilcoxon_q_bh"],
                    tp=row["paired_t_p"],
                    g=row["hedges_g_paired"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = collect(args.results_root, discover=args.discover)
    if frame.empty:
        raise SystemExit(f"no analysis outputs under {args.results_root}")
    frame.to_csv(args.output_dir / "summary_all.csv", index=False)

    subsets = collect_subsets(args.results_root)
    if not subsets.empty:
        subsets.to_csv(args.output_dir / "summary_subsets.csv", index=False)

    probe = collect_probe(args.results_root)
    probe_roll = probe_summary(probe)
    if not probe.empty:
        probe.to_csv(args.output_dir / "probe_all.csv", index=False)
    if not probe_roll.empty:
        probe_roll.to_csv(args.output_dir / "probe_summary.csv", index=False)

    seeds = collect_seed_variance(args.results_root, "esc")
    if not seeds.empty:
        seeds.to_csv(args.output_dir / "seed_variance.csv", index=False)

    lines = ["# GSE283614 measured-activity ablation - collected results", ""]
    lines.append(f"- comparisons: {frame.groupby(['state', 'comparison']).ngroups}")
    lines.append(f"- rows: {len(frame)}")
    lines.append("")
    lines.append(markdown_tables(frame))
    (args.output_dir / "summary_all.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
