from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse, spatial, stats
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.data import CONTROL_PATTERNS, extract_target_label, is_control_label, load_guide_activity_lookup
from perturbdiff.config import GuideFeatureEncoderConfig
from perturbdiff.guide_features import (
    build_notebook_cnn_feature_tensor,
    canonical_guide_keys,
    find_record_for_label,
    guide_cnn_feature_dim,
    load_guide_feature_records,
)
from perturbdiff.model import PerturbDiffusionModel

METRICS = ["mse", "e_distance", "pcc_delta", "wasserstein", "kl_divergence", "common_degs"]
HIGHER_IS_BETTER = {"mse": False, "e_distance": False, "pcc_delta": True, "wasserstein": False, "kl_divergence": False, "common_degs": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PerturbDiff checkpoints with the six benchmark metrics.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--perturbation-col", default="perturbation_label")
    parser.add_argument("--control-label", action="append", default=None, dest="control_labels")
    parser.add_argument("--min-test-cells-per-condition", type=int, default=1)
    parser.add_argument("--perturbdiff-checkpoint", type=Path, action="append", required=True, dest="checkpoints")
    parser.add_argument("--perturbdiff-name", action="append", required=True, dest="names")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="benchmark")
    parser.add_argument("--ddim-steps", type=int, default=5)
    parser.add_argument("--max-control-cells", type=int, default=128)
    parser.add_argument("--max-metric-cells", type=int, default=2000)
    parser.add_argument("--common-degs-method", choices=["auto", "fallback"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--activity-table", type=Path, default=None)
    parser.add_argument("--activity-col", default="relative_activity_day10")
    parser.add_argument("--guide-activity-cnn-table", type=Path, default=None)
    parser.add_argument("--guide-sequence-length", type=int, default=26)
    parser.add_argument("--eval-level", choices=["target", "guide"], default="guide")
    parser.add_argument("--target-strategy", choices=["first_token", "raw"], default="first_token")
    parser.add_argument("--no-change", action="store_true", default=False)
    parser.add_argument("--scgen", action="store_true", default=False)
    parser.add_argument("--scgen-model-dir", type=Path, default=None)
    parser.add_argument("--scgen-epochs", type=int, default=200)
    parser.add_argument("--scgen-batch-size", type=int, default=128)
    parser.add_argument("--scgen-n-hidden", type=int, default=800)
    parser.add_argument("--scgen-n-latent", type=int, default=100)
    parser.add_argument("--scgen-n-layers", type=int, default=2)
    parser.add_argument("--scgen-dropout-rate", type=float, default=0.2)
    return parser.parse_args()


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def vector_mean(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x).mean(axis=0)


def m_mse(true_mean: np.ndarray, pred_mean: np.ndarray) -> float:
    return float(np.mean((true_mean - pred_mean) ** 2))


def _limit_cells(x: np.ndarray, *, max_cells: int, seed: int) -> np.ndarray:
    if max_cells < 1 or x.shape[0] <= max_cells:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(x.shape[0], max_cells, replace=False)]


def m_edist(true_cells: np.ndarray, pred_cells: np.ndarray, *, max_cells: int, seed: int) -> float:
    x = _limit_cells(np.asarray(true_cells, np.float64), max_cells=max_cells, seed=seed)
    y = _limit_cells(np.asarray(pred_cells, np.float64), max_cells=max_cells, seed=seed + 1)
    cross = spatial.distance.cdist(x, y, "euclidean").mean()
    within_x = spatial.distance.pdist(x, "euclidean").mean() if x.shape[0] > 1 else 0.0
    within_y = spatial.distance.pdist(y, "euclidean").mean() if y.shape[0] > 1 else 0.0
    return float(2.0 * cross - within_x - within_y)


def m_pcc_delta(true_mean: np.ndarray, pred_mean: np.ndarray, control_mean: np.ndarray) -> float:
    true_delta = true_mean - control_mean
    pred_delta = pred_mean - control_mean
    if float(true_delta.std()) == 0.0 or float(pred_delta.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(true_delta, pred_delta)[0, 1])


def _pca_pair(x: np.ndarray, y: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, int]:
    n_components = min(n_components, x.shape[1], x.shape[0] + y.shape[0] - 1)
    if n_components < 1:
        return x, y, 0
    xy = np.vstack([x, y]).astype(np.float64, copy=False)
    xy = xy - xy.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    projected = xy @ vt[:n_components].T
    return projected[: x.shape[0]], projected[x.shape[0] :], n_components


def m_wasserstein(true_cells: np.ndarray, pred_cells: np.ndarray, *, pca_dim: int = 50) -> float:
    x, y, n_components = _pca_pair(np.asarray(true_cells, np.float64), np.asarray(pred_cells, np.float64), pca_dim)
    if n_components < 1:
        return 0.0
    return float(np.mean([stats.wasserstein_distance(x[:, dim], y[:, dim]) for dim in range(n_components)]))


def m_kl(true_cells: np.ndarray, pred_cells: np.ndarray, *, pca_dim: int = 50, bins: int = 50) -> float:
    x, y, n_components = _pca_pair(np.asarray(true_cells, np.float64), np.asarray(pred_cells, np.float64), pca_dim)
    if n_components < 1:
        return 0.0
    eps = 1e-12
    values = []
    for dim in range(n_components):
        xd = x[:, dim]
        yd = y[:, dim]
        lo = min(float(xd.min()), float(yd.min()))
        hi = max(float(xd.max()), float(yd.max()))
        if hi - lo < eps:
            values.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        px, _ = np.histogram(xd, bins=edges, density=True)
        py, _ = np.histogram(yd, bins=edges, density=True)
        px = px + eps
        py = py + eps
        px = px / px.sum()
        py = py / py.sum()
        values.append(0.5 * (stats.entropy(px, py) + stats.entropy(py, px)))
    return float(np.mean(values))


def _common_degs_fallback(true_cells: np.ndarray, pred_cells: np.ndarray, control_cells: np.ndarray, *, n_top: int) -> float:
    true_mean = np.mean(true_cells, axis=0)
    pred_mean = np.mean(pred_cells, axis=0)
    control_mean = np.mean(control_cells, axis=0)
    true_fc = np.abs(true_mean - control_mean) / (np.abs(control_mean) + 1e-8)
    pred_fc = np.abs(pred_mean - control_mean) / (np.abs(control_mean) + 1e-8)
    true_top = set(np.argsort(true_fc)[::-1][:n_top].tolist())
    pred_top = set(np.argsort(pred_fc)[::-1][:n_top].tolist())
    return float(len(true_top & pred_top) / max(len(true_top), 1))


def m_common_degs(
    true_cells: np.ndarray,
    pred_cells: np.ndarray,
    control_cells: np.ndarray,
    gene_names: list[str],
    *,
    method: str,
    n_top: int = 50,
    min_logfc: float = 0.25,
) -> float:
    if method == "fallback":
        return _common_degs_fallback(true_cells, pred_cells, control_cells, n_top=n_top)
    try:
        import scanpy as sc

        n_true = true_cells.shape[0]
        n_pred = pred_cells.shape[0]
        n_control = control_cells.shape[0]
        adata = sc.AnnData(np.vstack([true_cells, pred_cells, control_cells]))
        adata.var_names = gene_names
        adata.obs["group"] = ["treated"] * n_true + ["predicted"] * n_pred + ["control"] * n_control
        sc.pp.filter_genes(adata, min_cells=1)

        def degs(group: str, reference: str) -> set[str]:
            mask = adata.obs["group"].isin([group, reference])
            sub = adata[mask].copy()
            if sub.n_obs < 3:
                return set()
            sc.tl.rank_genes_groups(sub, groupby="group", groups=[group], reference=reference, method="t-test", n_genes=n_top * 2)
            df = sc.get.rank_genes_groups_df(sub, group=group)
            return set(df[df["logfoldchanges"].abs() >= min_logfc]["names"].iloc[:n_top])

        true_degs = degs("treated", "control")
        pred_degs = degs("predicted", "control")
        return float(len(true_degs & pred_degs) / max(len(true_degs), 1))
    except Exception:
        return _common_degs_fallback(true_cells, pred_cells, control_cells, n_top=n_top)


def evaluate_condition(
    true_cells: np.ndarray,
    pred_cells: np.ndarray,
    control_cells: np.ndarray,
    gene_names: list[str],
    *,
    max_metric_cells: int,
    common_degs_method: str,
    seed: int,
) -> dict[str, float]:
    true_mean = np.mean(true_cells, axis=0)
    pred_mean = np.mean(pred_cells, axis=0)
    control_mean = np.mean(control_cells, axis=0)
    true_limited = _limit_cells(true_cells, max_cells=max_metric_cells, seed=seed)
    pred_limited = _limit_cells(pred_cells, max_cells=max_metric_cells, seed=seed + 11)
    control_limited = _limit_cells(control_cells, max_cells=max_metric_cells, seed=seed + 23)
    return {
        "mse": m_mse(true_mean, pred_mean),
        "e_distance": m_edist(true_limited, pred_limited, max_cells=max_metric_cells, seed=seed),
        "pcc_delta": m_pcc_delta(true_mean, pred_mean, control_mean),
        "wasserstein": m_wasserstein(true_limited, pred_limited),
        "kl_divergence": m_kl(true_limited, pred_limited),
        "common_degs": m_common_degs(true_limited, pred_limited, control_limited, gene_names, method=common_degs_method),
    }


def compatible_config_from_state(state: dict) -> object:
    cfg = state["model_config"]
    state_dict = state["model_state_dict"]
    if not hasattr(cfg, "guide_encoder"):
        cfg = replace(cfg, guide_encoder=GuideFeatureEncoderConfig())
    if not hasattr(cfg, "control_reference"):
        # Legacy checkpoints never trained the counterfactual path at all, so they carry no
        # control-anchor convention.  Their inference always fed one control cell's latent per
        # sample, so "per_cell" reproduces their historical numbers exactly.
        cfg = replace(cfg, control_reference="per_cell")
    has_activity_condition = any(key.startswith("activity_condition.") for key in state_dict)
    if getattr(cfg.guide_encoder, "enabled", False):
        cfg = replace(
            cfg,
            guide_encoder=replace(
                cfg.guide_encoder,
                explicit_condition=has_activity_condition,
                activity_scales_delta=not has_activity_condition,
            ),
        )
    return cfg


def load_model(path: Path, device: torch.device) -> tuple[PerturbDiffusionModel, dict[str, int]]:
    state = torch.load(path, map_location=device, weights_only=False)
    model = PerturbDiffusionModel(compatible_config_from_state(state)).to(device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    mapping = {str(key): int(value) for key, value in state["perturbation_mapping"].items()}
    return model, mapping


def prepare_scgen_adata(adata, perturbation_col: str):
    prepared = adata.copy()
    prepared.obs["condition"] = prepared.obs[perturbation_col].astype(str)
    prepared.obs["cell_type"] = "all_cells"
    return prepared


def load_or_train_scgen(args: argparse.Namespace, train, perturbation_col: str):
    import scgen

    scgen_train = prepare_scgen_adata(train, perturbation_col)
    model_dir = args.scgen_model_dir
    if model_dir is not None and (model_dir / "model.pt").exists():
        model = scgen.SCGEN.load(str(model_dir), adata=scgen_train, accelerator="gpu" if args.device.startswith("cuda") else "cpu")
        return model, "loaded"

    scgen.SCGEN.setup_anndata(scgen_train, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(
        scgen_train,
        n_hidden=args.scgen_n_hidden,
        n_latent=args.scgen_n_latent,
        n_layers=args.scgen_n_layers,
        dropout_rate=args.scgen_dropout_rate,
    )
    train_kwargs = {
        "max_epochs": args.scgen_epochs,
        "batch_size": args.scgen_batch_size,
        "early_stopping": False,
    }
    if "use_gpu" in inspect.signature(model.train).parameters:
        train_kwargs["use_gpu"] = args.device.startswith("cuda")
    else:
        train_kwargs["accelerator"] = "gpu" if args.device.startswith("cuda") else "cpu"
        train_kwargs["devices"] = 1
    model.train(**train_kwargs)
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir), overwrite=True, save_anndata=False)
    return model, "trained"


def build_guide_features(labels: list[str], model: PerturbDiffusionModel, records_by_key: dict | None, sequence_length: int, activity_lookup: dict[str, float], device: torch.device) -> torch.Tensor | None:
    if getattr(model, "guide_doser", None) is None:
        return None
    feature_dim = int(model.config.guide_encoder.feature_dim or 0)
    if feature_dim < 1:
        return None
    if model.config.guide_encoder.activity_source == "direct":
        rows = np.zeros((len(labels), feature_dim), dtype=np.float32)
        for idx, label in enumerate(labels):
            for key in canonical_guide_keys(str(label)):
                value = activity_lookup.get(key)
                if value is not None:
                    rows[idx, 0] = float(value)
                    break
        return torch.from_numpy(rows).to(device)
    if records_by_key is None:
        raise ValueError("Predicted activity checkpoints require --guide-activity-cnn-table")
    expected_dim = guide_cnn_feature_dim(sequence_length=sequence_length)
    if feature_dim != expected_dim:
        raise ValueError(f"Checkpoint feature_dim={feature_dim} does not match sequence length feature_dim={expected_dim}")
    rows = []
    for label in labels:
        record = find_record_for_label(str(label), records_by_key)
        if record is None:
            rows.append(np.zeros(feature_dim, dtype=np.float32))
        else:
            rows.append(build_notebook_cnn_feature_tensor(record, sequence_length=sequence_length).reshape(-1))
    return torch.from_numpy(np.stack(rows, axis=0)).to(device)


@torch.no_grad()
def predict_model(
    model: PerturbDiffusionModel,
    mapping: dict[str, int],
    control_tensor: torch.Tensor,
    *,
    condition_label: str,
    target_label: str,
    eval_level: str,
    records_by_key: dict | None,
    activity_lookup: dict[str, float],
    sequence_length: int,
    steps: int,
) -> np.ndarray | None:
    candidates = [condition_label, target_label] if eval_level == "guide" else [target_label, condition_label]
    target_id = None
    for candidate in candidates:
        target_id = mapping.get(str(candidate))
        if target_id is not None:
            break
        fallback = extract_target_label(str(candidate), strategy="first_token")
        target_id = mapping.get(fallback)
        if target_id is not None:
            break
    if target_id is None:
        return None
    target_ids = torch.full((control_tensor.shape[0],), int(target_id), dtype=torch.long, device=control_tensor.device)
    feature_label = condition_label if eval_level == "guide" else target_label
    guide_features = build_guide_features(
        [feature_label] * control_tensor.shape[0],
        model,
        records_by_key,
        sequence_length,
        activity_lookup,
        control_tensor.device,
    )
    pred = model.sample_counterfactual(
        x_ctrl=control_tensor,
        target_ids=target_ids,
        guide_features=guide_features,
        steps=steps,
    )
    return pred.detach().cpu().numpy().astype(np.float64, copy=False)


def average(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def compute_summary(per_condition: list[dict[str, object]], method_names: list[str]) -> dict[str, object]:
    averages: dict[str, dict[str, float]] = {}
    for method in method_names:
        averages[method] = {}
        for metric in METRICS:
            values = [float(row[method][metric]) for row in per_condition if method in row and np.isfinite(float(row[method][metric]))]
            averages[method][metric] = average(values)

    ranks = {method: {metric: [] for metric in METRICS} for method in method_names}
    win_counts = {method: {metric: 0 for metric in METRICS} for method in method_names}
    for row in per_condition:
        for metric in METRICS:
            vals = []
            for method in method_names:
                if method not in row:
                    continue
                value = float(row[method][metric])
                if np.isfinite(value):
                    vals.append((method, value))
            if len(vals) < 2:
                continue
            vals.sort(key=lambda item: item[1], reverse=HIGHER_IS_BETTER[metric])
            win_counts[vals[0][0]][metric] += 1
            for rank, (method, _) in enumerate(vals, start=1):
                ranks[method][metric].append(float(rank))

    avg_ranks = {
        method: {metric: average(ranks[method][metric]) for metric in METRICS}
        for method in method_names
    }
    overall_rank = {
        method: average([value for value in avg_ranks[method].values() if np.isfinite(value)])
        for method in method_names
    }
    return {
        "n_conditions": len(per_condition),
        "method_averages": averages,
        "method_avg_ranks": avg_ranks,
        "method_metric_win_counts": win_counts,
        "method_overall_rank": overall_rank,
    }


def save_outputs(output_dir: Path, prefix: str, config: dict[str, object], summary: dict[str, object], per_condition: list[dict[str, object]], method_names: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"{prefix}_full.json"
    summary_path = output_dir / f"{prefix}_summary.csv"
    per_condition_path = output_dir / f"{prefix}_per_condition.csv"
    full_path.write_text(
        json.dumps({"config": config, "summary": summary, "per_condition": per_condition}, indent=2),
        encoding="utf-8",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method"] + METRICS + ["overall_rank"])
        for method in method_names:
            row = [method]
            for metric in METRICS:
                value = summary["method_averages"][method][metric]
                row.append(f"{value:.6f}" if np.isfinite(value) else "")
            rank = summary["method_overall_rank"][method]
            row.append(f"{rank:.3f}" if np.isfinite(rank) else "")
            writer.writerow(row)
    with per_condition_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "target_label", "n_test_cells"] + [f"{method}_{metric}" for method in method_names for metric in METRICS])
        for row in per_condition:
            out = [row["condition"], row["target_label"], row["n_test_cells"]]
            for method in method_names:
                values = row.get(method, {})
                for metric in METRICS:
                    value = values.get(metric, "")
                    out.append(f"{value:.6f}" if isinstance(value, float) and np.isfinite(value) else "")
            writer.writerow(out)
    print(f"Full -> {full_path}")
    print(f"Summary -> {summary_path}")
    print(f"Detail -> {per_condition_path}")


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) != len(args.names):
        raise ValueError("--perturbdiff-checkpoint and --perturbdiff-name must have the same length")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    control_labels = tuple(args.control_labels) if args.control_labels else CONTROL_PATTERNS

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("evaluate_benchmark requires anndata") from exc

    train = ad.read_h5ad(args.train_path)
    test = ad.read_h5ad(args.test_path)
    train_labels = train.obs[args.perturbation_col].astype(str).to_numpy()
    test_labels = test.obs[args.perturbation_col].astype(str).to_numpy()
    gene_names = list(test.var_names.astype(str))

    control_mask = np.array([is_control_label(label, control_labels=control_labels) for label in test_labels], dtype=bool)
    control_cells = to_dense(test[control_mask].X).astype(np.float64, copy=False)
    if control_cells.shape[0] == 0:
        raise ValueError("No control cells found in test split")
    control_indices = np.flatnonzero(control_mask)
    if args.max_control_cells > 0 and control_indices.size > args.max_control_cells:
        rng = np.random.default_rng(args.seed)
        control_indices = np.sort(rng.choice(control_indices, args.max_control_cells, replace=False))
    control_tensor_np = to_dense(test.X[control_indices]).astype(np.float32, copy=False)
    control_tensor = torch.from_numpy(control_tensor_np).to(torch.device(args.device))

    scgen_model = None
    scgen_status = None
    scgen_control = None
    if args.scgen:
        scgen_model, scgen_status = load_or_train_scgen(args, train, args.perturbation_col)
        scgen_control = test[control_indices].copy()
        scgen_control.obs["condition"] = control_labels[0] if control_labels else "control"
        scgen_control.obs["cell_type"] = "all_cells"

    train_set = set(str(label) for label in train_labels)
    train_targets = {
        extract_target_label(label, strategy=args.target_strategy)
        for label in train_set
        if not is_control_label(label, control_labels=control_labels)
    }
    counts = test.obs[args.perturbation_col].astype(str).value_counts()
    eval_items: list[dict[str, str]] = []
    for label, count in counts.items():
        label = str(label)
        if is_control_label(label, control_labels=control_labels) or label == "*":
            continue
        if int(count) < args.min_test_cells_per_condition:
            continue
        target = extract_target_label(label, strategy=args.target_strategy)
        if args.eval_level == "target":
            if label not in train_set:
                continue
            eval_items.append({"condition": label, "target_label": label})
        else:
            if target not in train_targets:
                continue
            eval_items.append({"condition": label, "target_label": target})
    eval_items = sorted(eval_items, key=lambda item: item["condition"])
    print(
        f"Evaluating {len(eval_items)} {args.eval_level} conditions, "
        f"generated controls={control_tensor.shape[0]}, metric controls={control_cells.shape[0]}",
        flush=True,
    )

    activity_lookup = (
        load_guide_activity_lookup(args.activity_table, activity_col=args.activity_col)
        if args.activity_table is not None
        else {}
    )
    records_by_key = (
        load_guide_feature_records([args.guide_activity_cnn_table])
        if args.guide_activity_cnn_table is not None
        else None
    )
    device = torch.device(args.device)
    models = {
        name: load_model(checkpoint, device)
        for name, checkpoint in zip(args.names, args.checkpoints)
    }
    method_names = list(args.names)
    if args.scgen:
        method_names.append("scgen")
    if args.no_change:
        method_names = ["no_change"] + method_names

    train_control_mask = np.array([is_control_label(label, control_labels=control_labels) for label in train_labels], dtype=bool)
    train_control_mean = vector_mean(train[train_control_mask].X).astype(np.float64, copy=False)

    per_condition: list[dict[str, object]] = []
    for idx, item in enumerate(eval_items, start=1):
        if idx == 1 or idx % 10 == 0:
            print(f"  [{idx}/{len(eval_items)}] {item['condition']}", flush=True)
        condition = item["condition"]
        target_label = item["target_label"]
        mask = test_labels == condition
        true_cells = to_dense(test[mask].X).astype(np.float64, copy=False)
        row: dict[str, object] = {
            "condition": condition,
            "target_label": target_label,
            "n_test_cells": int(true_cells.shape[0]),
            "n_control_cells": int(control_cells.shape[0]),
        }
        if args.no_change:
            pred = np.tile(train_control_mean, (true_cells.shape[0], 1))
            row["no_change"] = evaluate_condition(
                true_cells,
                pred,
                control_cells,
                gene_names,
                max_metric_cells=args.max_metric_cells,
                common_degs_method=args.common_degs_method,
                seed=args.seed + idx,
            )
        for name, (model, mapping) in models.items():
            pred = predict_model(
                model,
                mapping,
                control_tensor,
                condition_label=condition,
                target_label=target_label,
                eval_level=args.eval_level,
                records_by_key=records_by_key,
                activity_lookup=activity_lookup,
                sequence_length=args.guide_sequence_length,
                steps=args.ddim_steps,
            )
            if pred is None:
                continue
            row[name] = evaluate_condition(
                true_cells,
                pred,
                control_cells,
                gene_names,
                max_metric_cells=args.max_metric_cells,
                common_degs_method=args.common_degs_method,
                seed=args.seed + idx,
            )
        if scgen_model is not None:
            pred_scgen, _ = scgen_model.predict(
                ctrl_key=control_labels[0] if control_labels else "control",
                stim_key=condition,
                adata_to_predict=scgen_control,
            )
            pred_scgen_cells = to_dense(pred_scgen.X).astype(np.float64, copy=False)
            # Match the metric interface: each generated control is a cell.
            row["scgen"] = evaluate_condition(
                true_cells,
                pred_scgen_cells,
                control_cells,
                gene_names,
                max_metric_cells=args.max_metric_cells,
                common_degs_method=args.common_degs_method,
                seed=args.seed + idx,
            )
        per_condition.append(row)

    summary = compute_summary(per_condition, method_names)
    config = {
        "train_path": str(args.train_path),
        "test_path": str(args.test_path),
        "perturbation_col": args.perturbation_col,
        "eval_level": args.eval_level,
        "target_strategy": args.target_strategy,
        "min_test_cells_per_condition": args.min_test_cells_per_condition,
        "ddim_steps": args.ddim_steps,
        "max_control_cells": args.max_control_cells,
        "max_metric_cells": args.max_metric_cells,
        "common_degs_method": args.common_degs_method,
        "activity_table": None if args.activity_table is None else str(args.activity_table),
        "activity_col": args.activity_col,
        "device": args.device,
        "checkpoints": [str(path) for path in args.checkpoints],
        "names": args.names,
        "scgen": args.scgen,
        "scgen_model_dir": None if args.scgen_model_dir is None else str(args.scgen_model_dir),
        "scgen_model_status": scgen_status,
        "scgen_epochs": args.scgen_epochs,
        "scgen_batch_size": args.scgen_batch_size,
    }
    save_outputs(args.output_dir, args.output_prefix, config, summary, per_condition, method_names)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
