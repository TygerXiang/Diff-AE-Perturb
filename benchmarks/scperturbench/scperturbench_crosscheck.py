"""Cross-check our metric module against the literature code shipped in the image.

The image ships ``/home/project/Pertb_benchmark/manuscript2/calPerformance_genetic.py``,
whose ``calPerfor`` is the function that produced the paper's numbers.  This script

  1. writes our ground-truth DEG table into the layout ``getDEG`` expects
     (``DataSet2/<dataset>/DEG_hvg5000.pkl``),
  2. runs ``calPerfor`` on our ``result.h5ad`` for a subset of conditions and metrics,
  3. compares the result with our own ``score/per_condition.csv``.

Agreement (to float tolerance) is the evidence that our reimplementation reproduces the
published metric protocol rather than merely approximating it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")

import numpy as np
import pandas as pd

LITERATURE_ROOT = Path("/home/project/Pertb_benchmark")
LITERATURE_DATA = LITERATURE_ROOT / "DataSet2"

# our metric name -> (literature metric name, which of our columns holds the same quantity)
#
# ``calPerfor`` applies ``perf = np.log2(perf + 1)`` to ``sym_kldiv`` (lines 116-117 of
# ``calPerformance_genetic.py``), so the literature value corresponds to our ``value`` column
# (already log2-transformed), not to the raw pertpy ``raw_distance``.
# ``pearson_distance`` is returned by the literature without any post-processing and ours is
# reported as ``1 - perf``, so that one must be compared against ``raw_distance``.
METRIC_MAP = {
    "mse": ("mse", "value"),
    "pcc_delta": ("pearson_distance", "raw_distance"),
    "e_distance": ("edistance", "value"),
    "wasserstein": ("wasserstein", "value"),
    "kl_divergence": ("sym_kldiv", "value"),
}


def install_deg_table(deg_table_json: Path, dataset: str) -> Path:
    payload = json.loads(Path(deg_table_json).read_text(encoding="utf-8"))
    genes = payload["genes"] if "genes" in payload else payload
    out_dir = LITERATURE_DATA / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for condition, entry in genes.items():
        table[condition] = pd.DataFrame(
            {
                "pvals_adj": entry["pvals_adj"],
                "foldchanges": entry["foldchanges"],
                "scores": entry["scores"],
            },
            index=entry["order"],
        )
    target = out_dir / "DEG_hvg5000.pkl"
    with target.open("wb") as handle:
        pickle.dump(table, handle)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-check against the literature calPerfor.")
    parser.add_argument("--result-h5ad", type=Path, required=True)
    parser.add_argument("--deg-table", type=Path, required=True)
    parser.add_argument("--per-condition", type=Path, required=True)
    parser.add_argument("--method", default="crosscheck")
    parser.add_argument("--dataset", default="GSE132080")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="*", default=list(METRIC_MAP))
    parser.add_argument("--num-deg", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    import anndata as ad

    install_deg_table(args.deg_table, args.dataset)
    sys.path.insert(0, str(LITERATURE_ROOT))
    sys.path.insert(0, str(LITERATURE_ROOT / "manuscript2"))
    import calPerformance_genetic as literature

    adata = ad.read_h5ad(args.result_h5ad)
    ours = pd.read_csv(args.per_condition)

    conditions = args.conditions or sorted(ours["condition"].unique().tolist())
    comparisons = []
    for condition in conditions:
        for metric in args.metrics:
            lit_metric, column = METRIC_MAP[metric]
            result = literature.calPerfor(
                [adata, args.dataset, args.method, args.num_deg, args.seed, condition, "perturbation", "control", lit_metric]
            )
            if result is None:
                comparisons.append({"condition": condition, "metric": metric, "literature": None, "ours": None, "match": None})
                continue
            lit_value = float(result["performance"].iloc[0])
            row = ours[(ours["condition"] == condition) & (ours["metric"] == metric) & (ours["deg_budget"] == str(args.num_deg))]
            our_value = float(row[column].iloc[0]) if not row.empty else float("nan")
            # calPerfor rounds its output to 4 decimals, so agreement is checked against
            # half of the last rounded digit rather than float equality.
            diff = abs(lit_value - our_value) if np.isfinite(our_value) else float("inf")
            match = bool(diff <= 5.01e-5)
            comparisons.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "deg_budget": args.num_deg,
                    "literature": lit_value,
                    "ours": our_value,
                    "abs_diff": None if not np.isfinite(diff) else diff,
                    "match": match,
                }
            )
            print(f"{condition:8s} {metric:16s} lit={lit_value:.6f} ours={our_value:.6f} diff={diff:.2e} match={match}")

    matched = [c["match"] for c in comparisons if c["match"] is not None]
    summary = {
        "n_compared": len(matched),
        "n_match": int(sum(1 for m in matched if m)),
        "all_match": bool(matched) and all(matched),
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}, indent=2))
    return 0 if summary["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
