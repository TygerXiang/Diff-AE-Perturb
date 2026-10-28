"""Per-band analysis for the knockdown-stratum fold set.

The stratum design asks a sharper question than the pooled comparison: does the
activity channel help *because* the held-out set is a strength band the model never
trained on?  If the channel is really a strength proxy, its benefit should be
concentrated where strength is the thing that has to be inferred - the high band,
where guide identity inside the training range says nothing about how strong the
held-out guides are.

For every fold this reports the paired effect on each primary metric, plus the
band's knockdown range and how much activity information the band carries.  Fold
sizes are 10-12 guides, so these are descriptive - the pooled out-of-fold test is
still the headline - but the *pattern across bands* is the evidence this design
exists to produce.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
for candidate in (WORKSPACE, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

HIGHER_IS_BETTER = {
    "mse": False,
    "pcc_delta": True,
    "e_distance": False,
    "wasserstein": False,
    "kl_divergence": False,
    "common_degs": True,
}
PRIMARY = [
    ("mse", "100"),
    ("pcc_delta", "100"),
    ("e_distance", "100"),
    ("wasserstein", "100"),
    ("kl_divergence", "100"),
    ("common_degs", "cutoff"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-band paired analysis.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--dosage-table", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["activity_on", "activity_mask", "activity_perm"])
    parser.add_argument("--state", default="esc")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_arm(results_root: Path, state: str, arm: str, seed: int) -> pd.DataFrame:
    frames = []
    for fold_dir in sorted((results_root / "gse283614" / state).glob("fold_*")):
        path = fold_dir / arm / f"seed{seed}" / "score" / "per_condition.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["fold"] = int(fold_dir.name.split("_")[1])
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no scores for {arm}")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    conditions = json.loads(
        (args.inputs_dir / f"conditions_{args.state}.json").read_text(encoding="utf-8")
    )
    pc = conditions["per_condition"]
    dosage = pd.read_csv(args.dosage_table, sep="\t")
    dose = dict(zip(dosage["guide_identity"].astype(str), dosage["knockdown_fraction"]))
    target = dict(zip(dosage["guide_identity"].astype(str), dosage["dose_target"]))
    stratum = {str(f["fold"]): f.get("stratum", f"fold{f['fold']}") for f in conditions["folds"]}

    off = load_arm(args.results_root, args.state, "activity_off", args.seed)
    rows: list[dict] = []
    for arm in args.arms:
        try:
            arm_frame = load_arm(args.results_root, args.state, arm, args.seed)
        except SystemExit:
            print(f"  skip {arm}: no scores")
            continue
        merged = pd.concat([off.assign(_arm="off"), arm_frame.assign(_arm="on")], ignore_index=True)

        for fold in sorted({int(f["fold"]) for f in conditions["folds"]}):
            guides = [g for g in conditions["evaluated_conditions"] if pc[g]["fold"] == fold]
            band_guides = [g for g in guides if g in set(merged["condition"])]
            if len(band_guides) < 4:
                continue
            sub = merged[merged["condition"].isin(band_guides)]

            # Band provenance: what strength range this fold held out, and how
            # much activity information it carried.
            for t in ("NANOG", "POU5F1"):
                pass
            band_vals = sorted(dose[g] for g in band_guides)
            activity_vals = [pc[g]["activity_input"] for g in band_guides]
            n_nonzero = sum(1 for v in activity_vals if v > 0)

            for metric, budget in PRIMARY:
                cells = sub[(sub["metric"] == metric) & (sub["deg_budget"].astype(str) == budget)]
                wide = cells.pivot_table(index="condition", columns="_arm", values="value", aggfunc="mean").dropna()
                if wide.empty or "on" not in wide or "off" not in wide:
                    continue
                on = wide["on"].to_numpy(dtype=float)
                offv = wide["off"].to_numpy(dtype=float)
                diff = on - offv
                direction = 1 if HIGHER_IS_BETTER[metric] else -1
                try:
                    p_w = float(stats.wilcoxon(on, offv, zero_method="wilcox", alternative="two-sided").pvalue)
                except ValueError:
                    p_w = float("nan")
                rows.append(
                    {
                        "state": args.state,
                        "arm": arm,
                        "fold": fold,
                        "stratum": stratum[str(fold)],
                        "n_guides": len(band_guides),
                        "knockdown_min": band_vals[0],
                        "knockdown_max": band_vals[-1],
                        "n_nonzero_activity": n_nonzero,
                        "metric": metric,
                        "deg_budget": budget,
                        "mean_arm_on": float(on.mean()),
                        "mean_arm_off": float(offv.mean()),
                        "paired_delta": float(diff.mean()),
                        "relative_change_pct": float(100 * diff.mean() / offv.mean()) if offv.mean() else float("nan"),
                        "arm_on_better": bool(direction * diff.mean() > 0),
                        "wilcoxon_p": p_w,
                        "paired_t_p": float(stats.ttest_rel(on, offv).pvalue),
                    }
                )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    for arm in out["arm"].unique():
        print(f"\n=== {args.state.upper()} · {arm} vs activity_off · per band ===")
        sub = out[out["arm"] == arm]
        pivot = sub.pivot_table(
            index=["fold", "stratum", "n_guides", "n_nonzero_activity"],
            columns="metric",
            values="relative_change_pct",
        )
        print(pivot.to_string(float_format=lambda v: f"{v:+.1f}%"))
        print(" (positive % is not always better; see arm_on_better in the CSV)")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
