from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import torch
from scipy import sparse

import scgen

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.model import PerturbDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PerturbDiff and scGen on the same GSE132080 split."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "splits" / "gse132080_train.h5ad",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "splits" / "gse132080_test.h5ad",
    )
    parser.add_argument(
        "--perturbdiff-checkpoint",
        type=Path,
        default=REPO_ROOT / "runs" / "gse132080_with_split_1000ep" / "checkpoint_best.pt",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "runs" / "gse132080_scgen_compare" / "comparison_metrics.json",
    )
    parser.add_argument("--perturbation-col", default="perturbation_label")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--min-test-cells-per-target", type=int, default=20)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--scgen-epochs", type=int, default=200)
    parser.add_argument("--scgen-batch-size", type=int, default=128)
    parser.add_argument("--scgen-n-hidden", type=int, default=800)
    parser.add_argument("--scgen-n-latent", type=int, default=100)
    parser.add_argument("--scgen-n-layers", type=int, default=2)
    parser.add_argument("--scgen-dropout-rate", type=float, default=0.2)
    parser.add_argument("--signature-top-genes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def vector_mean(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x).mean(axis=0)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    if true_std == 0.0 or pred_std == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def evaluate_subset(
    y_true_mean: np.ndarray,
    y_pred_mean: np.ndarray,
    subset_indices: np.ndarray | None = None,
) -> dict[str, float]:
    if subset_indices is not None:
        y_true_mean = y_true_mean[subset_indices]
        y_pred_mean = y_pred_mean[subset_indices]
    r = pearson_r(y_true_mean, y_pred_mean)
    return {
        "pearson_r": r,
        "pearson_r2": float(r * r) if np.isfinite(r) else float("nan"),
        "mse": mse(y_true_mean, y_pred_mean),
    }


def build_target_signature(
    train_adata: ad.AnnData,
    perturbation_col: str,
    control_label: str,
    target: str,
    *,
    top_genes: int,
) -> dict[str, object] | None:
    train_labels = train_adata.obs[perturbation_col].astype(str).to_numpy()
    ctrl_mask = train_labels == control_label
    target_mask = train_labels == target
    if int(ctrl_mask.sum()) == 0 or int(target_mask.sum()) == 0:
        return None
    ctrl_mean = vector_mean(train_adata[ctrl_mask].X).astype(np.float64, copy=False)
    target_mean = vector_mean(train_adata[target_mask].X).astype(np.float64, copy=False)
    effect = target_mean - ctrl_mean
    candidate_indices = np.flatnonzero(np.abs(effect) > 0)
    if candidate_indices.size == 0:
        return None
    ranked = candidate_indices[np.argsort(np.abs(effect[candidate_indices]))[::-1]]
    signature_indices = ranked[: min(top_genes, ranked.size)]
    gene_names = np.asarray(train_adata.var_names.astype(str))
    target_matches = np.flatnonzero(gene_names == target)
    return {
        "signature_indices": signature_indices.astype(int),
        "signature_genes": gene_names[signature_indices].astype(str).tolist(),
        "target_gene_index": int(target_matches[0]) if target_matches.size else None,
    }


def evaluate_target(
    y_true_mean: np.ndarray,
    y_pred_mean: np.ndarray,
) -> dict[str, float]:
    return evaluate_subset(y_true_mean, y_pred_mean)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_adata = ad.read_h5ad(args.train_path)
    test_adata = ad.read_h5ad(args.test_path)
    if train_adata.n_vars != test_adata.n_vars or not np.array_equal(train_adata.var_names, test_adata.var_names):
        raise ValueError("train/test gene dimensions or var_names are not aligned")

    pert_col = args.perturbation_col
    ctrl = args.control_label
    test_ctrl = test_adata[test_adata.obs[pert_col].astype(str) == ctrl].copy()
    if test_ctrl.n_obs == 0:
        raise ValueError("No control cells found in test split")
    # scGen registry transfer requires the same obs keys as training AnnData.
    test_ctrl.obs["condition"] = ctrl
    test_ctrl.obs["cell_type"] = "all_cells"

    # 1) Train scGen on train split.
    scgen_train = train_adata.copy()
    scgen_train.obs["condition"] = scgen_train.obs[pert_col].astype(str)
    scgen_train.obs["cell_type"] = "all_cells"
    scgen.SCGEN.setup_anndata(scgen_train, batch_key="condition", labels_key="cell_type")
    scgen_model = scgen.SCGEN(
        scgen_train,
        n_hidden=args.scgen_n_hidden,
        n_latent=args.scgen_n_latent,
        n_layers=args.scgen_n_layers,
        dropout_rate=args.scgen_dropout_rate,
    )
    scgen_model.train(
        max_epochs=args.scgen_epochs,
        use_gpu=(args.device.startswith("cuda")),
        batch_size=args.scgen_batch_size,
        early_stopping=False,
    )

    # 2) Load PerturbDiff checkpoint.
    device = torch.device(args.device)
    state = torch.load(args.perturbdiff_checkpoint, map_location=device, weights_only=False)
    pd_model = PerturbDiffusionModel(state["model_config"]).to(device)
    pd_model.load_state_dict(state["model_state_dict"], strict=False)
    pd_model.eval()
    target_to_id: dict[str, int] = dict(state["perturbation_mapping"])
    train_ctrl_mask = train_adata.obs[pert_col].astype(str) == ctrl
    train_ctrl_mean = vector_mean(train_adata[train_ctrl_mask].X).astype(np.float64, copy=False)

    # 3) Shared evaluation targets.
    train_targets = set(train_adata.obs[pert_col].astype(str).unique())
    test_counts = test_adata.obs[pert_col].astype(str).value_counts()
    eval_targets = []
    for target, count in test_counts.items():
        if target == ctrl:
            continue
        if target not in train_targets:
            continue
        if target not in target_to_id:
            continue
        if count < args.min_test_cells_per_target:
            continue
        eval_targets.append(target)

    test_ctrl_np = to_dense(test_ctrl.X).astype(np.float32, copy=False)
    test_ctrl_tensor = torch.from_numpy(test_ctrl_np).to(device)

    per_target: list[dict[str, object]] = []
    for target in eval_targets:
        true_cells = test_adata[test_adata.obs[pert_col].astype(str) == target]
        y_true_mean = vector_mean(true_cells.X).astype(np.float64, copy=False)
        train_target_mean = vector_mean(
            train_adata[train_adata.obs[pert_col].astype(str) == target].X
        ).astype(np.float64, copy=False)
        signature = build_target_signature(
            train_adata,
            pert_col,
            ctrl,
            target,
            top_genes=args.signature_top_genes,
        )
        signature_indices = (
            None
            if signature is None
            else np.asarray(signature["signature_indices"], dtype=int)
        )

        pred_scgen, _ = scgen_model.predict(
            ctrl_key=ctrl,
            stim_key=target,
            adata_to_predict=test_ctrl,
        )
        y_scgen_mean = vector_mean(pred_scgen.X).astype(np.float64, copy=False)
        scgen_metrics = evaluate_target(y_true_mean, y_scgen_mean)

        with torch.no_grad():
            target_id = target_to_id[target]
            target_ids = torch.full((test_ctrl_tensor.shape[0],), target_id, dtype=torch.long, device=device)
            pred_pd = pd_model.sample_counterfactual(
                x_ctrl=test_ctrl_tensor,
                target_ids=target_ids,
                steps=args.ddim_steps,
            )
        y_pd_mean = pred_pd.mean(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
        pd_metrics = evaluate_target(y_true_mean, y_pd_mean)
        no_change_metrics = evaluate_target(y_true_mean, train_ctrl_mean)
        train_mean_metrics = evaluate_target(y_true_mean, train_target_mean)
        signature_metrics = {}
        if signature_indices is not None:
            signature_metrics = {
                "no_change": evaluate_subset(y_true_mean, train_ctrl_mean, signature_indices),
                "train_target_mean": evaluate_subset(y_true_mean, train_target_mean, signature_indices),
                "scgen": evaluate_subset(y_true_mean, y_scgen_mean, signature_indices),
                "perturbdiff": evaluate_subset(y_true_mean, y_pd_mean, signature_indices),
            }
        target_gene_index = None if signature is None else signature["target_gene_index"]

        per_target.append(
            {
                "target": target,
                "n_test_cells": int(true_cells.n_obs),
                "signature_genes": [] if signature is None else signature["signature_genes"],
                "target_gene_index": target_gene_index,
                "target_gene_true_mean": None
                if target_gene_index is None
                else float(y_true_mean[int(target_gene_index)]),
                "target_gene_no_change_mean": None
                if target_gene_index is None
                else float(train_ctrl_mean[int(target_gene_index)]),
                "target_gene_train_mean": None
                if target_gene_index is None
                else float(train_target_mean[int(target_gene_index)]),
                "target_gene_scgen_mean": None
                if target_gene_index is None
                else float(y_scgen_mean[int(target_gene_index)]),
                "target_gene_perturbdiff_mean": None
                if target_gene_index is None
                else float(y_pd_mean[int(target_gene_index)]),
                "no_change": no_change_metrics,
                "train_target_mean": train_mean_metrics,
                "scgen": scgen_metrics,
                "perturbdiff": pd_metrics,
                "signature_metrics": signature_metrics,
            }
        )
        print(
            f"{target}: "
            f"scGen R2={scgen_metrics['pearson_r2']:.4f}, "
            f"PerturbDiff R2={pd_metrics['pearson_r2']:.4f}, "
            f"train-mean R2={train_mean_metrics['pearson_r2']:.4f}"
        )

    def avg(metric_path: tuple[str, str]) -> float:
        values: list[float] = []
        for row in per_target:
            v = float(row[metric_path[0]][metric_path[1]])
            if np.isfinite(v):
                values.append(v)
        return float(np.mean(values)) if values else float("nan")

    def signature_avg(method: str, metric: str) -> float:
        values: list[float] = []
        for row in per_target:
            signature_metrics = row.get("signature_metrics", {})
            if method not in signature_metrics:
                continue
            v = float(signature_metrics[method][metric])
            if np.isfinite(v):
                values.append(v)
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "n_eval_targets": len(per_target),
        "signature_top_genes": args.signature_top_genes,
        "no_change_mean_pearson_r2": avg(("no_change", "pearson_r2")),
        "train_target_mean_pearson_r2": avg(("train_target_mean", "pearson_r2")),
        "scgen_mean_pearson_r2": avg(("scgen", "pearson_r2")),
        "perturbdiff_mean_pearson_r2": avg(("perturbdiff", "pearson_r2")),
        "no_change_mean_mse": avg(("no_change", "mse")),
        "train_target_mean_mse": avg(("train_target_mean", "mse")),
        "scgen_mean_mse": avg(("scgen", "mse")),
        "perturbdiff_mean_mse": avg(("perturbdiff", "mse")),
        "signature_metrics": {
            method: {
                "mean_pearson_r2": signature_avg(method, "pearson_r2"),
                "mean_mse": signature_avg(method, "mse"),
            }
            for method in ("no_change", "train_target_mean", "scgen", "perturbdiff")
        },
    }

    out = {
        "config": {
            "train_path": str(args.train_path),
            "test_path": str(args.test_path),
            "perturbdiff_checkpoint": str(args.perturbdiff_checkpoint),
            "perturbation_col": pert_col,
            "control_label": ctrl,
            "min_test_cells_per_target": args.min_test_cells_per_target,
            "ddim_steps": args.ddim_steps,
            "signature_top_genes": args.signature_top_genes,
            "scgen_epochs": args.scgen_epochs,
            "scgen_batch_size": args.scgen_batch_size,
            "device": args.device,
        },
        "summary": summary,
        "per_target": per_target,
    }
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved comparison to {output_path}")
    print(summary)


if __name__ == "__main__":
    main()
