"""Aggregate the T1.3/T1.4/T4.2/T4.3/T5.2 robustness re-scoring.

Reads the per-condition rows produced by ``tools/rescore_gse283614_de_vs_esc.py``
(one row per condition x metric x budget x variant x resample seed) and rebuilds the
paired on-vs-off comparison *inside each evaluation variant*, so the channel effect
is always measured against the off arm scored the same way.

Writes ``rescore_tests.csv`` (paired stats per variant/state/metric), and
``rescore_levels.csv`` (absolute metric levels per variant/state/metric/arm, useful
for checking that a variant did not simply move both arms).

Host-only: no metric is recomputed here, only re-aggregated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_gse283614_de_vs_esc as A  # noqa: E402  (reuse the frozen stat helpers)

SEEDED_VARIANTS = {"count_matched", "control_matched", "stratum_matched"}
HIGHER_IS_BETTER = {"mse": False, "pcc_delta": True, "e_distance": False,
                    "wasserstein": False, "kl_divergence": False, "common_degs": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "reports/assets/gse283614_de_vs_esc")
    parser.add_argument("--tag", default="kd3")
    args = parser.parse_args()
    src = args.root / f"rescore_{args.tag}" / "rescore_per_condition.csv"
    frame = pd.read_csv(src)
    frame["metric_deg"] = frame["metric"] + "@" + frame["deg_budget"].astype(str)
    frame = frame[frame["metric_deg"].isin([f"{m}@{b}" for m, b in A.ALL_CELLS])].copy()
    print(f"loaded {len(frame)} rows from {src}")

    # ---- absolute levels per variant/state/arm -------------------------------
    level = (frame.groupby(["variant", "state", "method", "metric", "deg_budget"],
                           as_index=False)
             .agg(value=("value", "mean"), n_imputed=("n_imputed", "mean"),
                  n_stimulated=("n_stimulated", "mean"), n_control=("n_control", "mean")))
    level.to_csv(args.root / f"rescore_levels_{args.tag}.csv", index=False)

    rows = []
    # one paired test per (variant, seed, state); the off arm is scored identically
    for variant, vblock in frame.groupby("variant"):
        seeds = sorted(vblock["resample_seed"].unique()) if variant in SEEDED_VARIANTS else \
            [int(vblock["resample_seed"].iloc[0])]
        for seed in seeds:
            sblock = vblock[vblock["resample_seed"] == seed]
            for state, stblock in sblock.groupby("state"):
                for (metric, budget), block in stblock.groupby(["metric", "deg_budget"], sort=False):
                    wide = block.pivot_table(index="condition", columns="method",
                                             values="value", aggfunc="mean")
                    if not {"activity_on", "activity_off"} <= set(wide.columns):
                        continue
                    pair = wide[["activity_on", "activity_off"]].dropna()
                    if len(pair) < 5:
                        continue
                    stat = A.paired_stats(
                        (pair["activity_on"] - pair["activity_off"]).to_numpy(dtype=float))
                    mean_on = float(pair["activity_on"].mean())
                    mean_off = float(pair["activity_off"].mean())
                    mean_abs_off = float(pair["activity_off"].abs().mean())
                    rel = A.relative_pct(mean_on, mean_off)
                    higher = HIGHER_IS_BETTER[metric]
                    rows.append({
                        "design": args.tag, "variant": variant, "resample_seed": int(seed),
                        "state": state, "metric": metric, "deg_budget": str(budget),
                        "n_conditions": stat["n"], "mean_on": mean_on, "mean_off": mean_off,
                        "relative_change_pct": rel,
                        "relative_change_pct_robust": A.relative_pct(mean_on, mean_abs_off),
                        "p_wilcoxon": stat["p_wilcoxon"], "p_t": stat["p_t"],
                        "hedges_g": stat["hedges_g"], "higher_is_better": bool(higher),
                        "better": bool((rel > 0) == higher) if np.isfinite(rel) else False,
                    })
    tests = pd.DataFrame(rows)
    primary = ["mse@100", "pcc_delta@100", "e_distance@100", "wasserstein@100",
               "kl_divergence@100", "common_degs@cutoff"]
    tests["_primary"] = (tests["metric"] + "@" + tests["deg_budget"]).isin(primary)
    tests["q_bh"] = np.nan
    for (variant, seed, state), block in tests.groupby(["variant", "resample_seed", "state"]):
        tests.loc[block.index, "q_bh"] = A.bh(block["p_wilcoxon"].to_numpy(dtype=float))
    order = {f"{m}@{b}": i for i, (m, b) in enumerate(A.ALL_CELLS)}
    tests["_order"] = (tests["metric"] + "@" + tests["deg_budget"]).map(order).fillna(99)
    tests = tests.sort_values(["variant", "state", "_order", "resample_seed"]).drop(
        columns=["_order", "_primary"])
    out = args.root / f"rescore_tests_{args.tag}.csv"
    tests.to_csv(out, index=False)
    print(f"wrote {out.name} ({len(tests)} rows)")

    # ---- headline: mean over resample seeds, primary metrics -----------------
    keep = tests[(tests["metric"] + "@" + tests["deg_budget"]).isin(primary)]
    head = keep.groupby(["variant", "state", "metric", "deg_budget"], as_index=False).agg(
        n_seeds=("resample_seed", "nunique"),
        rel_pct=("relative_change_pct", "mean"),
        rel_pct_min=("relative_change_pct", "min"),
        rel_pct_max=("relative_change_pct", "max"),
        rel_pct_robust=("relative_change_pct_robust", "mean"),
        p_min=("p_wilcoxon", "min"),
        q_min=("q_bh", "min"),
        n_conditions=("n_conditions", "max"))
    head["worse"] = np.where(head["metric"].isin(["pcc_delta", "common_degs"]),
                             head["rel_pct"] < 0, head["rel_pct"] > 0)
    head.to_csv(args.root / f"rescore_headline_{args.tag}.csv", index=False)
    print(f"wrote rescore_headline_{args.tag}.csv ({len(head)} rows)")
    print()
    print(head.pivot_table(index=["state", "metric", "deg_budget"], columns="variant",
                           values="rel_pct").to_string(float_format=lambda v: f"{v:+.1f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
