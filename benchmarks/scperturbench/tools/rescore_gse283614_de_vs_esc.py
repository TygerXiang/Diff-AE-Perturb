"""Re-score existing GSE283614 predictions under the pre-registered robustness protocols.

Nothing is retrained and no prediction is regenerated: this reads the
``result.h5ad`` files the published ablation already wrote and recomputes the six
metrics on modified *evaluation* sets, using ``scperturbench_metrics`` itself
(the same code path as the published numbers).

Variants (see ``.scperturbench/work/gse283614_de_vs_esc_predictions.md``):

  full              published protocol; must reproduce ``score/per_condition.csv``
  count_matched     T1.3  every condition subsampled to the design-wide minimum
                          of min(n_imputed, n_stimulated) in BOTH states
  control_matched   T1.4  control pool subsampled to the design-wide minimum
  stratum_matched   T4.3  imputed cells resampled so their state-score stratum
                          proportions match the ground truth's
  stratum_low/mid/high  T4.2  metrics computed inside one stratum only

Strata come from the paper's own gene sets (``state_score_genes.tsv``, exported by
``tools/analyze_gse283614_state_heterogeneity.py``): the DE score for DE cells and
the pluripotency score for ESC cells, split at the ground-truth tertiles.

Run inside the image's ``pertpyV7`` environment via
``.scperturbench/work/gse283614_de_vs_esc_rescore.sh``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import scperturbench_metrics as M

STRATA = ["low", "mid", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", default="gse283614")
    parser.add_argument("--states", nargs="+", default=["esc", "de"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=["activity_on", "activity_off"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resample-seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--variants", nargs="+", default=[
        "full", "count_matched", "control_matched", "stratum_matched",
        "stratum_low", "stratum_mid", "stratum_high"])
    parser.add_argument("--gene-sets", type=Path, required=True)
    parser.add_argument("--deg-prefix", default="deg_table")
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-conditions", type=int, default=0,
                        help="Smoke-test knob: score only the first N conditions.")
    return parser.parse_args()


def load_gene_sets(path: Path) -> dict[str, list[str]]:
    frame = pd.read_csv(path, sep="\t")
    return {state: block["gene"].astype(str).tolist()
            for state, block in frame.groupby("state")}


def state_score(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    index = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    idx = [index[g] for g in genes if g in index]
    if not idx:
        raise SystemExit("state-score gene set does not intersect the gene space")
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.asarray(X[:, idx], dtype=np.float64).mean(axis=1)


def rng_subsample(n: int, k: int, seed: int) -> np.ndarray:
    if k >= n:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=k, replace=False)


def take(adata: ad.AnnData, idx: np.ndarray) -> ad.AnnData:
    return adata[np.sort(idx)].copy()


def score_one(*, imputed, stimulated, control, condition, deg_table,
              method, resample_seed) -> list[dict]:
    rows = []
    for metric in M.METRICS:
        for budget in M.DEG_BUDGETS[metric]:
            scores = M.score_condition(
                imputed=imputed, stimulated=stimulated, control=control,
                condition=condition, deg_table=deg_table, metric=metric,
                num_deg=budget, seed=resample_seed)
            if scores is None:
                continue
            rows.append({
                "metric": metric, "deg_budget": M.budget_label(budget),
                "value": scores.get(metric), "raw_distance": scores.get("raw_distance"),
                "kl_floor": scores.get("kl_divergence_floor"),
                "n_imputed": scores.get("n_imputed"),
                "n_stimulated": scores.get("n_stimulated"),
                "n_control": scores.get("n_control"),
                "n_metric_genes": scores.get("n_metric_genes"),
            })
    for row in rows:
        row.update({"method": method, "condition": condition,
                    "resample_seed": int(resample_seed)})
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gene_sets = load_gene_sets(args.gene_sets)

    units = [(state, fold, arm) for state in args.states
             for fold in args.folds for arm in args.arms]

    # ---- pre-pass: the design-wide caps the two matched variants need --------
    caps = {"count": {}, "control": {}}
    per_unit_conditions: dict[tuple, list[str]] = {}
    for state, fold, arm in units:
        path = (args.results_root / args.dataset / state / f"fold_{fold}" / arm /
                f"seed{args.seed}" / "result.h5ad")
        if not path.exists():
            print(f"  missing {path}")
            continue
        result = ad.read_h5ad(path, backed="r")
        obs = result.obs
        control_n = int((obs[M.CATEGORY_COL].astype(str) == "control").sum())
        caps["control"][(state, fold, arm)] = control_n
        conditions = sorted(set(obs.loc[obs[M.CATEGORY_COL].astype(str) == "imputed",
                                       M.CONDITION_COL].astype(str)))
        if args.max_conditions:
            conditions = conditions[:args.max_conditions]
        per_unit_conditions[(state, fold, arm)] = conditions
        imputed_counts = obs.loc[obs[M.CATEGORY_COL].astype(str) == "imputed",
                                 M.CONDITION_COL].astype(str).value_counts()
        stimulated_counts = obs.loc[obs[M.CATEGORY_COL].astype(str) == "stimulated",
                                    M.CONDITION_COL].astype(str).value_counts()
        caps["count"][(state, fold, arm)] = {
            c: int(min(imputed_counts.get(c, 0), stimulated_counts.get(c, 0)))
            for c in conditions}
        del result

    count_cap = min(min(v.values()) for v in caps["count"].values() if v)
    control_cap = min(caps["control"].values())
    print(f"[caps] count={count_cap} control={control_cap}")
    (args.output_dir / "rescore_meta.json").write_text(json.dumps(
        {"count_cap": count_cap, "control_cap": control_cap,
         "resample_seeds": args.resample_seeds, "variants": args.variants,
         "result_units": sorted(f"{s}/fold_{f}/{a}" for s, f, a in units)},
        indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    for state, fold, arm in units:
        path = (args.results_root / args.dataset / state / f"fold_{fold}" / arm /
                f"seed{args.seed}" / "result.h5ad")
        if not path.exists():
            continue
        unit = f"{state}/fold_{fold}/{arm}"
        print(f"=== {unit} ===", flush=True)
        result = ad.read_h5ad(path)
        deg_table_path = args.inputs_dir / f"{args.deg_prefix}_{state}_fold{fold}.json"
        deg_table = M.load_deg_table(deg_table_path)

        control_all = result[result.obs[M.CATEGORY_COL].astype(str) == "control"].copy()
        conditions = per_unit_conditions[(state, fold, arm)]
        rows: list[dict] = []

        # ground-truth tertiles define the strata for this unit
        gt_all = result[result.obs[M.CATEGORY_COL].astype(str) == "stimulated"].copy()
        gt_score = state_score(gt_all, gene_sets[state])
        edges = np.quantile(gt_score, [1 / 3, 2 / 3])
        stratum_of = {cell: STRATA[s] for cell, s in
                      zip(gt_all.obs_names, np.digitize(gt_score, edges))}
        imputed_all = result[result.obs[M.CATEGORY_COL].astype(str) == "imputed"].copy()
        imp_score = state_score(imputed_all, gene_sets[state])
        imp_stratum = pd.Series(
            np.array(STRATA)[np.digitize(imp_score, edges)],
            index=imputed_all.obs_names.astype(str))
        gt_stratum = pd.Series(
            np.array(STRATA)[np.digitize(gt_score, edges)],
            index=gt_all.obs_names.astype(str))

        for condition in conditions:
            imputed = imputed_all[imputed_all.obs[M.CONDITION_COL].astype(str) == condition].copy()
            stimulated = gt_all[gt_all.obs[M.CONDITION_COL].astype(str) == condition].copy()
            if imputed.n_obs == 0 or stimulated.n_obs == 0:
                continue
            control = control_all

            def add(variant: str, seed: int, imp, stim, ctrl) -> None:
                for row in score_one(imputed=imp, stimulated=stim, control=ctrl,
                                     condition=condition, deg_table=deg_table,
                                     method=arm, resample_seed=seed):
                    row["variant"] = variant
                    rows.append(row)

            if "full" in args.variants:
                add("full", args.seed, imputed, stimulated, control)

            if "count_matched" in args.variants:
                for seed in args.resample_seeds:
                    imp = take(imputed, rng_subsample(imputed.n_obs, count_cap, seed))
                    stim = take(stimulated, rng_subsample(stimulated.n_obs, count_cap, seed + 100))
                    ctrl = take(control, rng_subsample(control.n_obs, count_cap, seed + 200))
                    add("count_matched", seed, imp, stim, ctrl)

            if "control_matched" in args.variants:
                for seed in args.resample_seeds:
                    ctrl = take(control, rng_subsample(control.n_obs, control_cap, seed + 300))
                    add("control_matched", seed, imputed, stimulated, ctrl)

            if "stratum_matched" in args.variants:
                gt_frac = (gt_stratum.reindex(stimulated.obs_names.astype(str))
                           .value_counts(normalize=True).reindex(STRATA).fillna(0.0))
                imp_lab = imp_stratum.reindex(imputed.obs_names.astype(str))
                total = int(imputed.n_obs)
                for seed in args.resample_seeds:
                    rng = np.random.default_rng(seed + 400)
                    chosen: list[str] = []
                    for stratum in STRATA:
                        pool = imp_lab.index[imp_lab == stratum].astype(str).tolist()
                        if not pool:
                            continue
                        # Target the ground truth's share of THIS condition's cells.
                        want = min(int(round(gt_frac[stratum] * total)), len(pool))
                        if want <= 0:
                            continue
                        picked = rng.choice(np.array(pool, dtype=object), size=want, replace=False)
                        chosen.extend(np.atleast_1d(picked).tolist())
                    if len(chosen) < 10:
                        continue
                    index = imputed.obs_names.astype(str)
                    keep = index.isin(chosen)
                    add("stratum_matched", seed, take(imputed, np.where(keep)[0]),
                        stimulated, control)

            for stratum in STRATA:
                variant = f"stratum_{stratum}"
                if variant not in args.variants:
                    continue
                imp_idx = np.where(imp_stratum.reindex(
                    imputed.obs_names.astype(str)).to_numpy() == stratum)[0]
                stim_idx = np.where(gt_stratum.reindex(
                    stimulated.obs_names.astype(str)).to_numpy() == stratum)[0]
                if len(imp_idx) < 5 or len(stim_idx) < 5:
                    continue
                ctrl_obs = control.obs_names.astype(str)
                ctrl_score = state_score(control, gene_sets[state])
                ctrl_idx = np.where(np.array(STRATA)[np.digitize(ctrl_score, edges)] == stratum)[0]
                if len(ctrl_idx) < 5:
                    continue
                add(variant, args.seed, take(imputed, imp_idx), take(stimulated, stim_idx),
                    take(control, ctrl_idx))

        frame = pd.DataFrame(rows)
        if frame.empty:
            print(f"  no rows for {unit}")
            continue
        frame.insert(0, "unit", unit)
        frame.insert(1, "state", state)
        frame.insert(2, "fold", fold)
        out = args.output_dir / f"rescore_{state}_fold{fold}_{arm}.csv"
        frame.to_csv(out, index=False)
        all_rows.extend(frame.to_dict("records"))
        print(f"  wrote {out.name} ({len(frame)} rows)", flush=True)

    combined = pd.DataFrame(all_rows)
    combined.to_csv(args.output_dir / "rescore_per_condition.csv", index=False)
    print(f"combined -> {args.output_dir / 'rescore_per_condition.csv'} ({len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
