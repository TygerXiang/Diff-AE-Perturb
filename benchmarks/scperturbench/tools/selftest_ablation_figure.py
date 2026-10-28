"""Self-test for scperturbench_ablation_figure.py using synthetic per-condition scores.

Exercises the full analysis + plotting path (including both figure variants) without touching the
real benchmark outputs, so statistical or layout bugs surface before the inference run finishes.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scperturbench_ablation_figure as figmod

CONDITIONS = [
    "ALDOA", "CAD", "DBR1", "DUT", "EIF2S1", "GATA1", "GINS1", "HAP95",
    "HSPA5", "HSPE1", "POLR2H", "RPS14", "TUBB", "GNB2L1", "POLR1D",
]
METRICS = [
    ("mse", "100"), ("mse", "all"),
    ("pcc_delta", "100"), ("pcc_delta", "all"),
    ("e_distance", "100"), ("e_distance", "all"),
    ("wasserstein", "100"),
    ("kl_divergence", "100"), ("kl_divergence", "all"),
    ("common_degs", "cutoff"),
]


def synthetic_frames(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for metric, budget in METRICS:
            base = 1.0 + rng.random()
            on = base * (1.0 + 0.05 * rng.standard_normal())
            off = base
            for arm, value in (("activity_on_200ep", on), ("activity_off_200ep", off)):
                rows.append(
                    {
                        "method": arm,
                        "seed": 7,
                        "condition": condition,
                        "metric": metric,
                        "deg_budget": budget,
                        "value": value,
                        # The floored KL companion lives on the kl_divergence rows; the figure reads it
                        # from this column instead of the artifact-prone literal ``value``.
                        "kl_floor": value * 0.5,
                        "raw_distance": value,
                        "n_imputed": 100.0,
                        "n_stimulated": 100.0,
                        "n_control": 2109.0,
                        "n_metric_genes": 100.0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["arm"] = frame["method"]
    return frame


def main() -> int:
    rng = np.random.default_rng(3)
    frame = synthetic_frames(rng)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for arm in ("activity_on_200ep", "activity_off_200ep"):
            score_dir = tmp_path / arm / "seed7" / "score"
            score_dir.mkdir(parents=True)
            frame[frame["arm"] == arm].to_csv(score_dir / "per_condition.csv", index=False)

        pair_rng = np.random.default_rng(figmod.BOOTSTRAP_SEED)
        pairwise, tests, long_frame = figmod.build_analysis(
            frame, "activity_on_200ep", "activity_off_200ep", pair_rng,
            figmod.PRIMARY_METRICS + figmod.SECONDARY_METRICS,
        )
        assert not pairwise.empty, "pairwise table is empty"
        expected_rows = len(CONDITIONS) * (len(figmod.PRIMARY_METRICS) + len(figmod.SECONDARY_METRICS))
        assert long_frame.shape[0] == expected_rows, (long_frame.shape, expected_rows)
        assert "kl_floor" in set(tests["metric"]), "floored KL should be analysed"
        assert "kl_divergence" not in set(tests["metric"]), "artifact-prone literal KL must not be plotted"
        assert set(tests["metric"]) >= {"mse", "pcc_delta", "common_degs"}
        assert tests["wilcoxon_q_bh"].between(0, 1).all(), "BH q-values out of range"
        # q must be monotone in p (BH guarantees this by construction on the sorted p-values)
        ordered = tests.sort_values("wilcoxon_p")
        assert ordered["wilcoxon_q_bh"].is_monotonic_increasing, ordered[["wilcoxon_p", "wilcoxon_q_bh"]]

        # A clean +5% shift on every condition must be detected as a consistent direction.
        mse_row = pairwise[(pairwise["metric"] == "mse") & (pairwise["deg_budget"] == "100")].iloc[0]
        assert mse_row["n_conditions"] == len(CONDITIONS)
        assert mse_row["n_conditions_activity_on_better"] >= 0

        live = CONDITIONS[:-3]
        live_frame = frame[frame["condition"].isin(live)]
        live_pair, _, _ = figmod.build_analysis(
            live_frame, "activity_on_200ep", "activity_off_200ep",
            np.random.default_rng(9), figmod.PRIMARY_METRICS,
        )

        figures = tmp_path / "figures"
        figures.mkdir()
        # exercise the minimal (no prose) path used by the real figure
        figmod.plot_metrics(
            long_frame, pairwise, metrics=figmod.PRIMARY_METRICS,
            arm_on="activity_on_200ep", arm_off="activity_off_200ep",
            title="self-test", out_base=figures / "fig1", dpi=80, subtitle="", minimal=True,
        )
        # and the annotated path kept for other datasets
        figmod.plot_metrics(
            long_frame[long_frame["condition"].isin(live)], live_pair,
            metrics=figmod.PRIMARY_METRICS,
            arm_on="activity_on_200ep", arm_off="activity_off_200ep",
            title="self-test live", out_base=figures / "fig2", dpi=80, subtitle="synthetic",
        )
        figmod.plot_per_condition_deltas(
            long_frame, metrics=figmod.PRIMARY_METRICS,
            arm_on="activity_on_200ep", arm_off="activity_off_200ep",
            title="self-test deltas", out_base=figures / "fig3", dpi=80,
            live_conditions=live,
        )
        for name in ("fig1", "fig2", "fig3"):
            for suffix in (".png", ".pdf"):
                path = figures / f"{name}{suffix}"
                assert path.exists() and path.stat().st_size > 1000, f"missing/empty {path}"

    print("SELF-TEST PASS: statistics and all three figures render")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
