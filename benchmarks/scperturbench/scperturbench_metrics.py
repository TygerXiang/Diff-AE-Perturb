"""scPerturBench-conformant six-metric scoring for the GSE132080 benchmark.

This module is a faithful port of the metric code shipped inside the
``localhost/scperturbench:squidiff`` image:

    /home/project/Pertb_benchmark/manuscript2/calPerformance_genetic.py
    /home/project/Pertb_benchmark/myUtil.py   (-> manuscript1/ood/myUtil.py)

and of the pertpy ``Distance`` metrics it relies on (pertpy 0.9.4,
``pertpy/tools/_distances/_distances.py``).

Literature primitive -> our metric name
---------------------------------------
    pertpy 'mse'               -> mse
    pertpy 'pearson_distance'  -> pcc_delta      (reported as 1 - distance, after control centering)
    pertpy 'edistance'         -> e_distance
    pertpy 'wasserstein'       -> wasserstein
    pertpy 'sym_kldiv'         -> kl_divergence  (log2(x + 1) applied, as in calPerfor)
    getDEG_cutoff-based F1     -> common_degs

Protocol reproduced verbatim from ``calPerfor``:
  * metrics are computed on a DEG-restricted gene subset (top-N by |t score|),
  * distribution metrics first subsample every group to <= 2000 cells with
    ``np.random.seed(42)`` (``f_subSample``),
  * ``pearson_distance`` is computed after ``calculateDelta`` (control centering),
  * ``sym_kldiv`` is ``log2(perf + 1)``,
  * wasserstein uses the 100-gene budget only.

Run inside the ``pertpyV7`` conda environment with ``JAX_PLATFORMS=cpu``.
"""

from __future__ import annotations

import os

# Must be set before numpy / jax are imported: the image's JAX CUDA plugin does not
# match the installed framework version, and pertpy's OTT solver is happy on CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Several scoring containers may run in parallel on the same machine; keep BLAS from
# oversubscribing the vCPU count. This only affects speed, never metric values.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import anndata as ad
import scanpy as sc

# --------------------------------------------------------------------------
# protocol constants
# --------------------------------------------------------------------------

METRICS = ["mse", "pcc_delta", "e_distance", "wasserstein", "kl_divergence", "common_degs"]

PERTPY_METRIC = {
    "mse": "mse",
    "pcc_delta": "pearson_distance",
    "e_distance": "edistance",
    "wasserstein": "wasserstein",
    "kl_divergence": "sym_kldiv",
}

HIGHER_IS_BETTER = {
    "mse": False,
    "pcc_delta": True,
    "e_distance": False,
    "wasserstein": False,
    "kl_divergence": False,
    "common_degs": True,
}

# calPerformance_genetic.py: numDEG_list = [100] for wasserstein, [100, 5000] otherwise.
# Our gene space is 1019 genes, so the 5000 budget means "all genes".
DEG_BUDGETS = {
    "mse": [100, -1],
    "pcc_delta": [100, -1],
    "e_distance": [100, -1],
    "wasserstein": [100],
    "kl_divergence": [100, -1],
    "common_degs": ["cutoff"],
}

SUBSAMPLED_METRICS = {"edistance", "wasserstein", "mean_var_distribution", "sym_kldiv"}
SUBSAMPLE_N = 2000
SUBSAMPLE_SEED = 42
DEG_CUTOFF_PADJ = 0.01
DEG_CUTOFF_FC_HI = 2.0
DEG_CUTOFF_FC_LO = 0.5
# Diagnostic std floor for the sym_kldiv companion column (the literature value uses 1e-8).
KL_STD_FLOOR = 0.1
CONTROL_TAG = "control"
CATEGORY_COL = "Expcategory"
CONDITION_COL = "perturbation"


def budget_label(budget) -> str:
    return "cutoff" if budget == "cutoff" else ("all" if int(budget) < 0 else str(int(budget)))


# --------------------------------------------------------------------------
# literature helpers (verbatim semantics)
# --------------------------------------------------------------------------


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def check_nan(adata: ad.AnnData, condition_column: str = CONDITION_COL, control_tag: str = CONTROL_TAG) -> ad.AnnData:
    """Port of checkNan(): impute NaN rows with the control mean."""
    adata1 = adata.copy()
    if sparse.issparse(adata1.X):
        adata1.X = adata1.X.toarray()
    nan_rows = np.where(np.isnan(adata1.X).any(axis=1))[0]
    if len(nan_rows) >= 1:
        a = adata1[adata1.obs[condition_column] == control_tag].X.mean(axis=0)
        a = np.asarray(a).reshape([1, -1])
        adata1.X[nan_rows] = np.tile(a, [len(nan_rows), 1])
    return adata1


def calculate_delta(adata: ad.AnnData, control_tag: str = CONTROL_TAG) -> ad.AnnData:
    """Port of calculateDelta(): subtract the control mean, concat [control, imputed, stimulated]."""
    control = adata[adata.obs[CATEGORY_COL] == control_tag].copy()
    imputed = adata[adata.obs[CATEGORY_COL] == "imputed"].copy()
    stimulated = adata[adata.obs[CATEGORY_COL] == "stimulated"].copy()
    control_mean = np.asarray(control.X.mean(axis=0)).ravel()
    imputed.X = np.asarray(imputed.X) - control_mean
    stimulated.X = np.asarray(stimulated.X) - control_mean
    return ad.concat([control, imputed, stimulated])


def f_sub_sample(adata: ad.AnnData, n_samples: int = SUBSAMPLE_N, control_tag: str = CONTROL_TAG) -> ad.AnnData:
    """Port of f_subSample(): seeded subsample of imputed/control/stimulated, concat in that order."""

    def sub_sample(a: ad.AnnData, n: int) -> ad.AnnData:
        if a.shape[0] <= n:
            return a
        idx = np.random.choice(a.n_obs, n, replace=False)
        return a[idx, :]

    np.random.seed(SUBSAMPLE_SEED)
    imputed = sub_sample(adata[adata.obs[CATEGORY_COL] == "imputed"], n_samples)
    control = sub_sample(adata[adata.obs[CATEGORY_COL] == control_tag], n_samples)
    stimulated = sub_sample(adata[adata.obs[CATEGORY_COL] == "stimulated"], n_samples)
    return ad.concat([imputed, control, stimulated])


def _pertpy_distance(adata: ad.AnnData, metric: str) -> float:
    import pertpy as pt

    with _SuppressOutput():
        distance = pt.tools.Distance(metric=metric, layer_key="X")
        pairwise = distance.onesided_distances(
            adata,
            groupby=CATEGORY_COL,
            selected_group="imputed",
            groups=["stimulated"],
            show_progressbar=False,
        )
    return float(pairwise["stimulated"])


def sym_kldiv_with_floor(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-8, std_floor: float = 0.0) -> float:
    """Same formula as pertpy ``SymmetricKLDivergence`` with an optional std floor.

    The literature uses ``epsilon = 1e-8``; genes whose standard deviation is
    effectively zero in either group then dominate the mean (2**40-scale values),
    which is not informative. The floored value is reported only as a diagnostic
    next to the literal metric, never in its place.
    """
    values = []
    for index in range(x.shape[1]):
        x_std = max(float(x[:, index].std()) + epsilon, std_floor)
        y_std = max(float(y[:, index].std()) + epsilon, std_floor)
        x_mean = float(x[:, index].mean())
        y_mean = float(y[:, index].mean())
        kl = np.log(y_std / x_std) + (x_std**2 + (x_mean - y_mean) ** 2) / (2 * y_std**2) - 0.5
        klr = np.log(x_std / y_std) + (y_std**2 + (y_mean - x_mean) ** 2) / (2 * x_std**2) - 0.5
        values.append(kl + klr)
    return float(np.mean(values))


class _SuppressOutput:
    """Port of the literature's SuppressOutput context manager."""

    def __enter__(self):
        import sys

        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        import sys

        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        return False


# --------------------------------------------------------------------------
# DEG handling
# --------------------------------------------------------------------------


def load_deg_table(path: Path) -> dict:
    """Load a DEG table produced by build-inputs (literature DEG_hvg5000.pkl layout)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["genes"] if "genes" in payload else payload


def deg_list(deg_table: dict, condition: str, num_deg) -> list[str]:
    entry = deg_table.get(condition)
    if not entry:
        return []
    genes = entry["order"]
    if num_deg == "cutoff" or int(num_deg) < 0:
        return list(genes)
    return list(genes[: int(num_deg)])


def deg_cutoff_set(deg_table: dict, condition: str) -> set[str]:
    """Port of getDEG_cutoff(): pvals_adj <= 0.01 and (FC >= 2 or FC <= 0.5)."""
    entry = deg_table.get(condition)
    if not entry:
        return set()
    selected = []
    for name, padj, fc in zip(entry["order"], entry["pvals_adj"], entry["foldchanges"]):
        if padj is None or fc is None:
            continue
        if padj <= DEG_CUTOFF_PADJ and (fc >= DEG_CUTOFF_FC_HI or fc <= DEG_CUTOFF_FC_LO):
            selected.append(name)
    return set(selected)


def rank_genes_groups_table(adata: ad.AnnData, group: str, reference: str = CONTROL_TAG, condition_col: str = CONDITION_COL) -> pd.DataFrame:
    """Reuse the literature's calDEG recipe: t-test + 2**logfoldchanges, sorted by |scores|."""
    adata = adata.copy()
    if "log1p" not in adata.uns:
        adata.uns["log1p"] = {"base": None}
    sub = adata[adata.obs[condition_col].isin([group, reference])].copy()
    sc.tl.rank_genes_groups(sub, condition_col, groups=[group], reference=reference, method="t-test")
    result = sub.uns["rank_genes_groups"]
    frame = pd.DataFrame(
        {
            "names": result["names"][group],
            "pvals_adj": result["pvals_adj"][group],
            "logfoldchanges": result["logfoldchanges"][group],
            "scores": result["scores"][group],
        }
    )
    frame["foldchanges"] = 2 ** frame["logfoldchanges"]
    frame = frame.drop(columns=["logfoldchanges"]).set_index("names")
    frame["abs_scores"] = np.abs(frame["scores"])
    frame = frame.sort_values("abs_scores", ascending=False)
    frame = frame[~frame.index.duplicated()]
    return frame


def common_degs_metric(
    imputed: ad.AnnData,
    control: ad.AnnData,
    gt_degs: set[str],
    *,
    condition_col: str = CONDITION_COL,
) -> dict[str, float]:
    """DEG recovery (F1) using the literature's getDEG_cutoff definition on both sides.

    The published image defines ``getDEG_cutoff`` (ground-truth DEG set) but does not
    ship the aggregation function, so we report precision / recall / F1 / Jaccard and
    the raw set sizes for the same cutoff applied to the predicted (imputed) cells.
    """
    if not gt_degs:
        return {
            "common_degs": float("nan"),
            "deg_precision": float("nan"),
            "deg_recall": float("nan"),
            "deg_jaccard": float("nan"),
            "n_gt_degs": 0.0,
            "n_pred_degs": float("nan"),
            "n_deg_overlap": float("nan"),
        }
    combined = ad.concat(
        [
            imputed.copy(),
            control.copy(),
        ]
    )
    combined.obs[condition_col] = ["imputed"] * imputed.n_obs + [CONTROL_TAG] * control.n_obs
    try:
        frame = rank_genes_groups_table(combined, "imputed", reference=CONTROL_TAG, condition_col=condition_col)
        pred = {
            name
            for name, padj, fc in zip(frame.index, frame["pvals_adj"], frame["foldchanges"])
            if padj is not None and fc is not None and padj <= DEG_CUTOFF_PADJ and (fc >= DEG_CUTOFF_FC_HI or fc <= DEG_CUTOFF_FC_LO)
        }
    except Exception:
        pred = set()
    overlap = len(gt_degs & pred)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(gt_degs)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    jaccard = overlap / len(gt_degs | pred) if (gt_degs | pred) else 0.0
    return {
        "common_degs": float(f1),
        "deg_precision": float(precision),
        "deg_recall": float(recall),
        "deg_jaccard": float(jaccard),
        "n_gt_degs": float(len(gt_degs)),
        "n_pred_degs": float(len(pred)),
        "n_deg_overlap": float(overlap),
    }


# --------------------------------------------------------------------------
# per-condition scoring (port of calPerfor)
# --------------------------------------------------------------------------


def score_condition(
    *,
    imputed: ad.AnnData,
    stimulated: ad.AnnData,
    control: ad.AnnData,
    condition: str,
    deg_table: dict,
    metric: str,
    num_deg,
    seed: int,
    do_subsample: bool = True,
    control_tag: str = CONTROL_TAG,
) -> dict[str, float] | None:
    if imputed.n_obs == 0 or stimulated.n_obs == 0:
        return None

    if metric == "common_degs":
        gt = deg_cutoff_set(deg_table, condition)
        result = common_degs_metric(imputed, control, gt)
        result["n_imputed"] = float(imputed.n_obs)
        result["n_stimulated"] = float(stimulated.n_obs)
        result["n_control"] = float(control.n_obs)
        return result

    pertpy_metric = PERTPY_METRIC[metric]
    adata = ad.concat([stimulated, imputed, control])
    adata.obs[CONDITION_COL] = [condition] * stimulated.n_obs + [condition] * imputed.n_obs + [control_tag] * control.n_obs
    adata.obs[CATEGORY_COL] = ["stimulated"] * stimulated.n_obs + ["imputed"] * imputed.n_obs + ["control"] * control.n_obs

    genes = deg_list(deg_table, condition, num_deg)
    genes = [g for g in genes if g in adata.var_names]
    if not genes:
        return None
    adata = adata[:, genes].copy()
    adata = check_nan(adata, CONDITION_COL, control_tag)

    if pertpy_metric == "pearson_distance":
        adata = calculate_delta(adata, control_tag)
    adata.layers["X"] = adata.X.copy()

    if do_subsample and pertpy_metric in SUBSAMPLED_METRICS:
        adata = f_sub_sample(adata, SUBSAMPLE_N, control_tag)

    try:
        perf = _pertpy_distance(adata, pertpy_metric)
    except Exception as exc:  # noqa: BLE001 - mirror the literature's try/except
        print(f"    [warn] {condition} {metric} n={budget_label(num_deg)} failed: {exc}")
        perf = float("nan")

    raw = float(perf)
    if pertpy_metric == "sym_kldiv":
        perf = np.log2(perf + 1)

    value = float(perf)
    if metric == "pcc_delta":
        value = 1.0 - value
    result = {
        metric: value,
        "raw_distance": raw,  # pertpy value before the literature's post-processing
        "n_imputed": float(imputed.n_obs),
        "n_stimulated": float(stimulated.n_obs),
        "n_control": float(control.n_obs),
        "n_metric_genes": float(len(genes)),
    }
    if metric == "kl_divergence":
        imp = np.asarray(adata[adata.obs[CATEGORY_COL] == "imputed"].layers["X"], dtype=np.float64)
        stim = np.asarray(adata[adata.obs[CATEGORY_COL] == "stimulated"].layers["X"], dtype=np.float64)
        floored = np.log2(sym_kldiv_with_floor(imp, stim, std_floor=KL_STD_FLOOR) + 1)
        result["kl_divergence_floor"] = float(floored)
    return result


# --------------------------------------------------------------------------
# result h5ad IO contract
# --------------------------------------------------------------------------


def write_result(path: Path, adata: ad.AnnData, *, method: str, seed: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = adata.copy()
    out.obs["method"] = method
    out.obs["seed"] = int(seed)
    out.uns["benchmark_meta"] = json.dumps(meta, ensure_ascii=False)
    out.write_h5ad(path)


def validate_result(adata: ad.AnnData, *, expected_genes: list[str]) -> list[str]:
    problems = []
    genes = [str(g) for g in adata.var_names]
    if genes != list(expected_genes):
        problems.append(f"gene order/space mismatch (got {len(genes)}, expected {len(expected_genes)})")
    for column in (CONDITION_COL, CATEGORY_COL):
        if column not in adata.obs:
            problems.append(f"missing obs column {column!r}")
    if problems:
        return problems
    categories = set(adata.obs[CATEGORY_COL].astype(str))
    if not {"control", "imputed", "stimulated"} <= categories:
        problems.append(f"Expcategory must contain control/imputed/stimulated, got {sorted(categories)}")
    X = to_dense(adata.X)
    if not np.isfinite(X).all():
        problems.append("non-finite values in X")
    return problems


# --------------------------------------------------------------------------
# scoring driver
# --------------------------------------------------------------------------


def summarize(rows: list[dict], method: str) -> dict:
    per_metric: dict[str, list[float]] = {}
    for row in rows:
        key = f"{row['metric']}@{row['deg_budget']}"
        value = row["value"]
        if value is None or not np.isfinite(value):
            continue
        per_metric.setdefault(key, []).append(float(value))
        if row.get("kl_floor") is not None and np.isfinite(row["kl_floor"]):
            per_metric.setdefault(f"kl_divergence_floor@{row['deg_budget']}", []).append(float(row["kl_floor"]))
    return {key: float(np.mean(values)) for key, values in per_metric.items() if values}


def score_result(
    *,
    result_path: Path,
    test_path: Path,
    deg_table: dict,
    output_dir: Path,
    method: str,
    seed: int,
    conditions: list[str] | None = None,
    max_conditions: int = 0,
) -> dict:
    import pertpy as pt  # noqa: F401 - fail fast if the env is wrong

    _ = pt.__version__
    result = ad.read_h5ad(result_path)
    test = ad.read_h5ad(test_path)

    genes = [str(g) for g in test.var_names]
    problems = validate_result(result, expected_genes=genes)
    if problems:
        raise SystemExit(f"result.h5ad contract violation: {problems}")

    if conditions is None:
        conditions = sorted(
            set(result.obs.loc[result.obs[CATEGORY_COL].astype(str) == "imputed", CONDITION_COL].astype(str))
        )
    if max_conditions > 0:
        conditions = conditions[:max_conditions]

    control = result[result.obs[CATEGORY_COL].astype(str) == "control"].copy()
    rows: list[dict] = []
    for index, condition in enumerate(conditions, start=1):
        imputed = result[(result.obs[CATEGORY_COL].astype(str) == "imputed") & (result.obs[CONDITION_COL].astype(str) == condition)].copy()
        stimulated = result[(result.obs[CATEGORY_COL].astype(str) == "stimulated") & (result.obs[CONDITION_COL].astype(str) == condition)].copy()
        print(f"  [{index}/{len(conditions)}] {method} {condition} (imputed={imputed.n_obs}, stimulated={stimulated.n_obs})", flush=True)
        for metric in METRICS:
            for budget in DEG_BUDGETS[metric]:
                scores = score_condition(
                    imputed=imputed,
                    stimulated=stimulated,
                    control=control,
                    condition=condition,
                    deg_table=deg_table,
                    metric=metric,
                    num_deg=budget,
                    seed=seed,
                )
                if scores is None:
                    continue
                rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "condition": condition,
                        "metric": metric,
                        "deg_budget": budget_label(budget),
                        "value": scores.get(metric),
                        "raw_distance": scores.get("raw_distance"),
                        "kl_floor": scores.get("kl_divergence_floor"),
                        "n_imputed": scores.get("n_imputed"),
                        "n_stimulated": scores.get("n_stimulated"),
                        "n_control": scores.get("n_control"),
                        "n_metric_genes": scores.get("n_metric_genes"),
                        "n_gt_degs": scores.get("n_gt_degs"),
                        "n_pred_degs": scores.get("n_pred_degs"),
                        "n_deg_overlap": scores.get("n_deg_overlap"),
                        "deg_precision": scores.get("deg_precision"),
                        "deg_recall": scores.get("deg_recall"),
                        "deg_jaccard": scores.get("deg_jaccard"),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    per_condition_path = output_dir / "per_condition.csv"
    frame.to_csv(per_condition_path, index=False)

    summary = summarize(rows, method)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "seed", "metric_budget", "value"])
        for key in sorted(summary):
            writer.writerow([method, int(seed), key, f"{summary[key]:.6f}"])

    full = {
        "method": method,
        "seed": int(seed),
        "result_path": str(result_path),
        "test_path": str(test_path),
        "n_conditions": len(conditions),
        "conditions": list(conditions),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "not_conformant_note": (
            "Metrics follow scPerturBench (pertpy Distance + DEG restriction + seeded subsampling). "
            "The DEG aggregation is not shipped in the image; we use getDEG_cutoff sets with F1."
        ),
        "per_condition": rows,
    }
    (output_dir / "full.json").write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary -> {summary_path}")
    return full


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one result.h5ad with the scPerturBench six metrics.")
    parser.add_argument("--result-h5ad", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--deg-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--max-conditions", type=int, default=0)
    args = parser.parse_args()

    deg_table = load_deg_table(args.deg_table)
    score_result(
        result_path=args.result_h5ad,
        test_path=args.test_path,
        deg_table=deg_table,
        output_dir=args.output_dir,
        method=args.method,
        seed=args.seed,
        conditions=args.conditions,
        max_conditions=args.max_conditions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
