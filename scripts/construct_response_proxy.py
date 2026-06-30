from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a PS-inspired response_proxy column for train/val/test AnnData splits. "
            "The proxy is fit on the train split only, then applied to all splits."
        )
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "splits" / "gse132080_train.h5ad",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "splits" / "gse132080_val.h5ad",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "splits" / "gse132080_test.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "splits_response_proxy",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT
        / "reproducibility"
        / "prepared"
        / "splits_response_proxy"
        / "response_proxy_summary.json",
    )
    parser.add_argument("--perturbation-col", default="perturbation_label")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--response-proxy-col", default="response_proxy")
    parser.add_argument("--top-genes", type=int, default=50)
    parser.add_argument("--min-target-cells", type=int, default=20)
    parser.add_argument("--min-abs-effect", type=float, default=1e-6)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    parser.add_argument(
        "--exclude-perturbed-gene",
        action="store_true",
        help="Exclude the target gene itself from that target's signature if it is present.",
    )
    return parser.parse_args()


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def vector_mean(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x).mean(axis=0)


def robust_minmax(values: np.ndarray, lower_q: float, upper_q: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(finite, lower_q))
    hi = float(np.quantile(finite, upper_q))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def build_target_models(
    train: ad.AnnData,
    *,
    perturbation_col: str,
    control_label: str,
    top_genes: int,
    min_target_cells: int,
    min_abs_effect: float,
    exclude_perturbed_gene: bool,
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    labels = train.obs[perturbation_col].astype(str).to_numpy()
    ctrl_mask = labels == control_label
    if int(ctrl_mask.sum()) == 0:
        raise ValueError(f"No control cells found with label {control_label!r}")

    control_mean = vector_mean(train[ctrl_mask].X).astype(np.float64, copy=False)
    gene_names = np.asarray(train.var_names.astype(str))
    models: dict[str, dict[str, object]] = {}

    for target in sorted(set(labels) - {control_label}):
        target_mask = labels == target
        if int(target_mask.sum()) < min_target_cells:
            continue

        target_mean = vector_mean(train[target_mask].X).astype(np.float64, copy=False)
        effect = target_mean - control_mean
        candidate_mask = np.abs(effect) >= min_abs_effect
        if exclude_perturbed_gene:
            candidate_mask &= gene_names != target
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            continue

        ranked = candidate_indices[np.argsort(np.abs(effect[candidate_indices]))[::-1]]
        signature_indices = ranked[: min(top_genes, ranked.size)]
        signature_effect = effect[signature_indices]
        norm = float(np.dot(signature_effect, signature_effect))
        if norm <= 0:
            continue
        models[str(target)] = {
            "signature_indices": signature_indices.astype(int).tolist(),
            "signature_genes": gene_names[signature_indices].astype(str).tolist(),
            "effect": signature_effect.astype(float).tolist(),
            "n_train_cells": int(target_mask.sum()),
            "mean_abs_effect": float(np.mean(np.abs(signature_effect))),
            "effect_norm": norm,
            "target_gene_in_signature": bool(target in set(gene_names[signature_indices])),
        }
    return control_mean, models


def raw_projection_scores(
    adata: ad.AnnData,
    control_mean: np.ndarray,
    model: dict[str, object],
    *,
    ridge: float,
) -> np.ndarray:
    idx = np.asarray(model["signature_indices"], dtype=int)
    effect = np.asarray(model["effect"], dtype=np.float64)
    x = to_dense(adata[:, idx].X).astype(np.float64, copy=False)
    centered = x - control_mean[idx][None, :]
    denom = float(np.dot(effect, effect) + ridge)
    return centered @ effect / denom


def score_split(
    adata: ad.AnnData,
    train_scalers: dict[str, tuple[float, float]],
    control_mean: np.ndarray,
    models: dict[str, dict[str, object]],
    *,
    perturbation_col: str,
    control_label: str,
    response_proxy_col: str,
    ridge: float,
) -> ad.AnnData:
    out = adata.copy()
    labels = out.obs[perturbation_col].astype(str).to_numpy()
    proxy = np.zeros(out.n_obs, dtype=np.float32)
    raw_proxy = np.zeros(out.n_obs, dtype=np.float32)
    usable = np.zeros(out.n_obs, dtype=bool)

    for target, model in models.items():
        mask = labels == target
        if not mask.any():
            continue
        raw = raw_projection_scores(out[mask], control_mean, model, ridge=ridge)
        lo, hi = train_scalers[target]
        scaled = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
        proxy[mask] = scaled.astype(np.float32)
        raw_proxy[mask] = raw.astype(np.float32)
        usable[mask] = True

    control_mask = labels == control_label
    usable[control_mask] = True
    out.obs[response_proxy_col] = proxy
    out.obs[f"{response_proxy_col}_raw"] = raw_proxy
    out.obs[f"{response_proxy_col}_usable"] = usable
    out.uns["response_proxy"] = {
        "response_proxy_col": response_proxy_col,
        "method": "train_fit_signature_projection",
        "description": (
            "PS-inspired scalar response proxy. Target-specific mean effect directions "
            "are fit on the train split versus controls. Each cell is projected onto its "
            "target direction and robustly scaled to [0, 1] using train target cells."
        ),
    }
    return out


def summarize_split(adata: ad.AnnData, response_proxy_col: str, perturbation_col: str) -> dict[str, object]:
    proxy = adata.obs[response_proxy_col].astype(float).to_numpy()
    usable = adata.obs[f"{response_proxy_col}_usable"].astype(bool).to_numpy()
    labels = adata.obs[perturbation_col].astype(str)
    by_target: dict[str, dict[str, float | int]] = {}
    for target, count in labels.value_counts().items():
        mask = labels.to_numpy() == str(target)
        vals = proxy[mask]
        by_target[str(target)] = {
            "n_cells": int(count),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p05": float(np.quantile(vals, 0.05)),
            "p50": float(np.quantile(vals, 0.50)),
            "p95": float(np.quantile(vals, 0.95)),
        }
    return {
        "n_cells": int(adata.n_obs),
        "usable_fraction": float(np.mean(usable)),
        "mean": float(np.mean(proxy)),
        "std": float(np.std(proxy)),
        "p05": float(np.quantile(proxy, 0.05)),
        "p50": float(np.quantile(proxy, 0.50)),
        "p95": float(np.quantile(proxy, 0.95)),
        "by_target": by_target,
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.lower_quantile < args.upper_quantile <= 1:
        raise ValueError("Expected 0 <= lower_quantile < upper_quantile <= 1")

    train = ad.read_h5ad(args.train_path)
    val = ad.read_h5ad(args.val_path)
    test = ad.read_h5ad(args.test_path)
    for split_name, split in {"val": val, "test": test}.items():
        if not np.array_equal(train.var_names, split.var_names):
            raise ValueError(f"train and {split_name} var_names are not aligned")

    control_mean, models = build_target_models(
        train,
        perturbation_col=args.perturbation_col,
        control_label=args.control_label,
        top_genes=args.top_genes,
        min_target_cells=args.min_target_cells,
        min_abs_effect=args.min_abs_effect,
        exclude_perturbed_gene=args.exclude_perturbed_gene,
    )
    if not models:
        raise ValueError("No target response models were constructed")

    train_scalers: dict[str, tuple[float, float]] = {}
    train_labels = train.obs[args.perturbation_col].astype(str).to_numpy()
    for target, model in models.items():
        mask = train_labels == target
        raw = raw_projection_scores(train[mask], control_mean, model, ridge=args.ridge)
        train_scalers[target] = robust_minmax(raw, args.lower_quantile, args.upper_quantile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    split_inputs = {
        "train": train,
        "val": val,
        "test": test,
    }
    split_outputs = {}
    for split_name, split in split_inputs.items():
        scored = score_split(
            split,
            train_scalers,
            control_mean,
            models,
            perturbation_col=args.perturbation_col,
            control_label=args.control_label,
            response_proxy_col=args.response_proxy_col,
            ridge=args.ridge,
        )
        output_path = args.output_dir / f"gse132080_{split_name}.h5ad"
        scored.write_h5ad(output_path)
        split_outputs[split_name] = {
            "path": str(output_path),
            "summary": summarize_split(scored, args.response_proxy_col, args.perturbation_col),
        }

    summary = {
        "method": "PS-inspired train-fit signature projection",
        "reference": "Li and colleagues, Decoding heterogeneous single-cell perturbation responses, Nature Cell Biology 2025, DOI 10.1038/s41556-025-01626-9",
        "parameters": {
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "test_path": str(args.test_path),
            "perturbation_col": args.perturbation_col,
            "control_label": args.control_label,
            "response_proxy_col": args.response_proxy_col,
            "top_genes": args.top_genes,
            "min_target_cells": args.min_target_cells,
            "min_abs_effect": args.min_abs_effect,
            "ridge": args.ridge,
            "lower_quantile": args.lower_quantile,
            "upper_quantile": args.upper_quantile,
            "exclude_perturbed_gene": args.exclude_perturbed_gene,
        },
        "n_targets_modeled": len(models),
        "target_models": {
            target: {
                key: value
                for key, value in model.items()
                if key not in {"signature_indices", "effect"}
            }
            | {
                "scaler_low": train_scalers[target][0],
                "scaler_high": train_scalers[target][1],
            }
            for target, model in models.items()
        },
        "splits": split_outputs,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"constructed response_proxy for {len(models)} targets")
    print(f"wrote splits to {args.output_dir}")
    print(f"summary: {args.summary_json}")


if __name__ == "__main__":
    main()
