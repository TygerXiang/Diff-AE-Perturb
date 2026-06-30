from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate sgRNA virtual strength assignments against the original "
            "Perturb-seq activity table and expression-derived response_proxy summaries."
        )
    )
    parser.add_argument(
        "--virtual-scales-csv",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "gse132080_sgrna_200ep"
        / "guide_scale_export"
        / "guide_virtual_scales_128_sgrnas.csv",
    )
    parser.add_argument(
        "--activity-table",
        type=Path,
        default=Path(
            r"C:\Users\Admin\myprojects\diff-project\datasets\GSE132080"
            r"\41587_2019_387_MOESM3_ESM\Table_S16_perturb-seq_screen_phenotypes.txt"
        ),
    )
    parser.add_argument(
        "--response-proxy-train",
        type=Path,
        default=Path(
            r"C:\Users\Admin\myprojects\diff-project\reproducibility\prepared"
            r"\splits_response_proxy\gse132080_train.h5ad"
        ),
    )
    parser.add_argument(
        "--response-proxy-val",
        type=Path,
        default=Path(
            r"C:\Users\Admin\myprojects\diff-project\reproducibility\prepared"
            r"\splits_response_proxy\gse132080_val.h5ad"
        ),
    )
    parser.add_argument(
        "--response-proxy-test",
        type=Path,
        default=Path(
            r"C:\Users\Admin\myprojects\diff-project\reproducibility\prepared"
            r"\splits_response_proxy\gse132080_test.h5ad"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gse132080_sgrna_200ep" / "guide_strength_validation",
    )
    return parser.parse_args()


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> dict[str, float | int | None]:
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    x_vals = x.to_numpy()[mask]
    y_vals = y.to_numpy()[mask]
    if x_vals.size < 3:
        return {"n": int(x_vals.size), "stat": None, "pvalue": None}
    if np.allclose(x_vals, x_vals[0]) or np.allclose(y_vals, y_vals[0]):
        return {"n": int(x_vals.size), "stat": None, "pvalue": None}
    if method == "pearson":
        stat, pvalue = pearsonr(x_vals, y_vals)
    elif method == "spearman":
        stat, pvalue = spearmanr(x_vals, y_vals)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return {"n": int(x_vals.size), "stat": float(stat), "pvalue": float(pvalue)}


def pairwise_rank_agreement(
    df: pd.DataFrame,
    *,
    group_col: str,
    x_col: str,
    y_col: str,
) -> dict[str, float | int | None]:
    concordant = 0
    discordant = 0
    ties = 0
    used_groups = 0
    total_pairs = 0
    for _, group in df.groupby(group_col):
        sub = group[[x_col, y_col]].dropna()
        if len(sub) < 2:
            continue
        used_groups += 1
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        for i in range(len(sub) - 1):
            dx = x[i + 1 :] - x[i]
            dy = y[i + 1 :] - y[i]
            pair_mask = (dx != 0) & (dy != 0)
            total_pairs += int(pair_mask.sum())
            if not pair_mask.any():
                ties += len(dx)
                continue
            signs = np.sign(dx[pair_mask] * dy[pair_mask])
            concordant += int((signs > 0).sum())
            discordant += int((signs < 0).sum())
            ties += int((dx == 0).sum() + (dy == 0).sum())
    usable_pairs = concordant + discordant
    agreement = None if usable_pairs == 0 else concordant / usable_pairs
    kendall_like = None if usable_pairs == 0 else (concordant - discordant) / usable_pairs
    return {
        "groups_used": int(used_groups),
        "concordant_pairs": int(concordant),
        "discordant_pairs": int(discordant),
        "usable_pairs": int(usable_pairs),
        "ignored_ties": int(ties),
        "agreement_fraction": None if agreement is None else float(agreement),
        "kendall_like": None if kendall_like is None else float(kendall_like),
    }


def add_relative_to_matched(
    df: pd.DataFrame,
    *,
    value_col: str,
    target_col: str = "target",
    matched_col_name: str | None = None,
    ratio_col_name: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if matched_col_name is None:
        matched_col_name = f"{value_col}_matched"
    if ratio_col_name is None:
        ratio_col_name = f"{value_col}_ratio_to_matched"
    matched_values = (
        out.loc[out["is_fully_matched"], [target_col, value_col]]
        .dropna()
        .drop_duplicates(subset=[target_col])
        .set_index(target_col)[value_col]
    )
    out[matched_col_name] = out[target_col].map(matched_values)
    out[ratio_col_name] = out[value_col] / out[matched_col_name]
    return out


def load_virtual_and_activity(virtual_path: Path, activity_path: Path) -> pd.DataFrame:
    virtual = pd.read_csv(virtual_path)
    virtual["is_fully_matched"] = virtual["id"] == virtual["fully_matched_id"]
    virtual["activity_key"] = virtual["target"].astype(str) + "|" + virtual["sequence"].astype(str)

    activity = pd.read_csv(activity_path, sep="\t")
    first_col = activity.columns[0]
    activity = activity.rename(columns={first_col: "activity_label", "gene": "target"})
    activity["activity_key"] = activity["target"].astype(str) + "|" + activity["sequence"].astype(str)
    activity["relative_activity_mean"] = activity[
        ["relative_activity_day5", "relative_activity_day10"]
    ].mean(axis=1)
    activity["gamma_mean"] = activity[["gamma_day5", "gamma_day10"]].mean(axis=1)

    merged = virtual.merge(
        activity[
            [
                "activity_key",
                "activity_label",
                "gamma_day5",
                "gamma_day10",
                "gamma_mean",
                "relative_activity_day5",
                "relative_activity_day10",
                "relative_activity_mean",
            ]
        ],
        on="activity_key",
        how="left",
        validate="one_to_one",
    )
    return merged.drop(columns=["activity_key"])


def summarize_response_proxy(path: Path, split_name: str) -> pd.DataFrame:
    adata = ad.read_h5ad(path)
    obs = adata.obs.copy()
    obs["guide_identity"] = obs["guide_identity"].astype(str)
    obs["perturbation_label"] = obs["perturbation_label"].astype(str)
    if "response_proxy_usable" in obs:
        obs = obs.loc[obs["response_proxy_usable"].astype(bool)]
    obs = obs.loc[
        (obs["perturbation_label"] != "control")
        & (obs["perturbation_label"] != "*")
        & (obs["guide_identity"] != "*")
    ].copy()
    grouped = (
        obs.groupby("guide_identity", dropna=False)
        .agg(
            n_cells=("response_proxy", "size"),
            response_proxy_mean=("response_proxy", "mean"),
            response_proxy_median=("response_proxy", "median"),
            response_proxy_std=("response_proxy", "std"),
            response_proxy_raw_mean=("response_proxy_raw", "mean"),
            target=("perturbation_label", lambda s: s.iloc[0]),
        )
        .reset_index()
        .rename(columns={"guide_identity": "label"})
    )
    grouped["split"] = split_name
    return grouped


def attach_proxy_summaries(merged: pd.DataFrame, proxy_tables: list[pd.DataFrame]) -> pd.DataFrame:
    out = merged.copy()
    for table in proxy_tables:
        split_name = str(table["split"].iloc[0])
        renamed = table.drop(columns=["split", "target"]).rename(
            columns={
                "n_cells": f"{split_name}_proxy_n_cells",
                "response_proxy_mean": f"{split_name}_proxy_mean",
                "response_proxy_median": f"{split_name}_proxy_median",
                "response_proxy_std": f"{split_name}_proxy_std",
                "response_proxy_raw_mean": f"{split_name}_proxy_raw_mean",
            }
        )
        out = out.merge(renamed, on="label", how="left", validate="one_to_one")
        out = add_relative_to_matched(
            out,
            value_col=f"{split_name}_proxy_mean",
            matched_col_name=f"{split_name}_proxy_mean_matched",
            ratio_col_name=f"{split_name}_proxy_ratio",
        )
        out = add_relative_to_matched(
            out,
            value_col=f"{split_name}_proxy_median",
            matched_col_name=f"{split_name}_proxy_median_matched",
            ratio_col_name=f"{split_name}_proxy_median_ratio",
        )
    return out


def pooled_proxy_table(proxy_tables: list[pd.DataFrame]) -> pd.DataFrame:
    pooled = pd.concat(proxy_tables, ignore_index=True)
    summary = (
        pooled.groupby("label", dropna=False)
        .agg(
            pooled_proxy_n_cells=("n_cells", "sum"),
            pooled_proxy_mean=("response_proxy_mean", "mean"),
            pooled_proxy_median=("response_proxy_median", "mean"),
            pooled_proxy_raw_mean=("response_proxy_raw_mean", "mean"),
            target=("target", lambda s: s.iloc[0]),
        )
        .reset_index()
    )
    return summary


def correlation_block(
    df: pd.DataFrame,
    predictors: list[str],
    outcomes: list[str],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for predictor in predictors:
        out[predictor] = {}
        for outcome in outcomes:
            out[predictor][outcome] = {
                "pearson": safe_corr(df[predictor], df[outcome], "pearson"),
                "spearman": safe_corr(df[predictor], df[outcome], "spearman"),
            }
    return out


def build_summary(merged: pd.DataFrame) -> dict[str, object]:
    predictors = ["s", "test_proxy_ratio", "pooled_proxy_ratio"]
    outcomes = ["relative_activity_day5", "relative_activity_day10", "relative_activity_mean"]

    mismatched = merged.loc[~merged["is_fully_matched"]].copy()
    mismatched_gt1 = mismatched.loc[mismatched["s"] > 1].copy()

    summary = {
        "n_guides": int(len(merged)),
        "n_targets": int(merged["target"].nunique()),
        "n_fully_matched_guides": int(merged["is_fully_matched"].sum()),
        "n_mismatched_guides": int((~merged["is_fully_matched"]).sum()),
        "n_mismatched_virtual_gt1": int(len(mismatched_gt1)),
        "correlations_vs_activity": correlation_block(merged, predictors, outcomes),
        "correlations_virtual_vs_proxy": {
            "s_vs_test_proxy_ratio": {
                "pearson": safe_corr(merged["s"], merged["test_proxy_ratio"], "pearson"),
                "spearman": safe_corr(merged["s"], merged["test_proxy_ratio"], "spearman"),
            },
            "s_vs_pooled_proxy_ratio": {
                "pearson": safe_corr(merged["s"], merged["pooled_proxy_ratio"], "pearson"),
                "spearman": safe_corr(merged["s"], merged["pooled_proxy_ratio"], "spearman"),
            },
        },
        "within_target_pairwise_agreement": {
            "virtual_s_vs_activity_mean": pairwise_rank_agreement(
                merged, group_col="target", x_col="s", y_col="relative_activity_mean"
            ),
            "test_proxy_ratio_vs_activity_mean": pairwise_rank_agreement(
                merged, group_col="target", x_col="test_proxy_ratio", y_col="relative_activity_mean"
            ),
            "pooled_proxy_ratio_vs_activity_mean": pairwise_rank_agreement(
                merged, group_col="target", x_col="pooled_proxy_ratio", y_col="relative_activity_mean"
            ),
            "virtual_s_vs_test_proxy_ratio": pairwise_rank_agreement(
                merged, group_col="target", x_col="s", y_col="test_proxy_ratio"
            ),
        },
        "mismatched_virtual_gt1_truth_check": {
            "count": int(len(mismatched_gt1)),
            "fraction_activity_day5_gt1": float((mismatched_gt1["relative_activity_day5"] > 1).mean()),
            "fraction_activity_day10_gt1": float((mismatched_gt1["relative_activity_day10"] > 1).mean()),
            "fraction_activity_mean_gt1": float((mismatched_gt1["relative_activity_mean"] > 1).mean()),
            "fraction_test_proxy_ratio_gt1": float((mismatched_gt1["test_proxy_ratio"] > 1).mean()),
            "fraction_pooled_proxy_ratio_gt1": float((mismatched_gt1["pooled_proxy_ratio"] > 1).mean()),
        },
        "mismatched_virtual_le1_truth_check": {
            "count": int((mismatched["s"] <= 1).sum()),
            "fraction_activity_mean_gt1": float((mismatched.loc[mismatched["s"] <= 1, "relative_activity_mean"] > 1).mean()),
            "fraction_test_proxy_ratio_gt1": float((mismatched.loc[mismatched["s"] <= 1, "test_proxy_ratio"] > 1).mean()),
            "fraction_pooled_proxy_ratio_gt1": float((mismatched.loc[mismatched["s"] <= 1, "pooled_proxy_ratio"] > 1).mean()),
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_virtual_and_activity(args.virtual_scales_csv, args.activity_table)
    if merged["activity_label"].isna().any():
        missing = merged.loc[merged["activity_label"].isna(), ["label", "target", "sequence"]]
        raise ValueError(f"Missing activity annotations for guides:\n{missing.to_string(index=False)}")

    proxy_tables = [
        summarize_response_proxy(args.response_proxy_train, "train"),
        summarize_response_proxy(args.response_proxy_val, "val"),
        summarize_response_proxy(args.response_proxy_test, "test"),
    ]
    pooled = pooled_proxy_table(proxy_tables)
    merged = attach_proxy_summaries(merged, proxy_tables)
    merged = merged.merge(
        pooled.drop(columns=["target"]),
        on="label",
        how="left",
        validate="one_to_one",
    )
    merged = add_relative_to_matched(
        merged,
        value_col="pooled_proxy_mean",
        matched_col_name="pooled_proxy_mean_matched",
        ratio_col_name="pooled_proxy_ratio",
    )
    merged = add_relative_to_matched(
        merged,
        value_col="pooled_proxy_median",
        matched_col_name="pooled_proxy_median_matched",
        ratio_col_name="pooled_proxy_median_ratio",
    )

    merged["virtual_minus_activity_mean"] = merged["s"] - merged["relative_activity_mean"]
    merged["abs_virtual_minus_activity_mean"] = merged["virtual_minus_activity_mean"].abs()
    merged["is_activity_mean_gt1"] = merged["relative_activity_mean"] > 1
    merged["is_test_proxy_ratio_gt1"] = merged["test_proxy_ratio"] > 1
    merged["is_pooled_proxy_ratio_gt1"] = merged["pooled_proxy_ratio"] > 1

    summary = build_summary(merged)

    mismatched_gt1 = merged.loc[(~merged["is_fully_matched"]) & (merged["s"] > 1)].copy()
    top_disagreements = merged.sort_values("abs_virtual_minus_activity_mean", ascending=False).head(20)

    merged_path = args.output_dir / "guide_strength_validation_merged.csv"
    mismatched_path = args.output_dir / "mismatched_virtual_gt1_cases.csv"
    disagreements_path = args.output_dir / "largest_virtual_activity_disagreements.csv"
    summary_path = args.output_dir / "summary.json"

    merged.to_csv(merged_path, index=False)
    mismatched_gt1.to_csv(mismatched_path, index=False)
    top_disagreements.to_csv(disagreements_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"merged table: {merged_path}")
    print(f"summary json: {summary_path}")
    print(f"mismatched s>1 cases: {mismatched_path}")
    print(f"top disagreements: {disagreements_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
