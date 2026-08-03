from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff import (
    DiffusionConfig,
    GuideFeatureEncoderConfig,
    ModelConfig,
    PerturbDiffusionModel,
    TrainingLossConfig,
    build_virtual_guide_scales_from_embedding,
)
from perturbdiff.data import (
    CONTROL_PATTERNS,
    build_signature_gene_weights,
    load_guide_activity_lookup,
    load_h5ad_perturbation_dataset,
)
from perturbdiff.guide_features import (
    build_notebook_cnn_feature_tensor,
    canonical_guide_keys,
    find_record_for_label,
    guide_cnn_feature_dim,
    load_guide_feature_records,
)


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train simplified conditional DDIM PerturbDiff.")
    parser.add_argument("--data-path", required=True, help="Prepared .h5ad input.")
    parser.add_argument("--perturbation-col", default="perturbation_label")
    parser.add_argument(
        "--activity-label-col",
        default=None,
        help="Optional obs column used to join guide-level activity independently of perturbation_col.",
    )
    parser.add_argument("--target-strategy", choices=("first_token", "raw"), default="raw")
    parser.add_argument("--control-label", action="append", dest="control_labels")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--train-path", default=None, help="Optional pre-split train .h5ad.")
    parser.add_argument("--val-path", default=None, help="Optional pre-split val .h5ad.")
    parser.add_argument("--test-path", default=None, help="Optional pre-split test .h5ad.")
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--encoder-hidden-dims", default="512,256")
    parser.add_argument("--denoiser-hidden-dims", default="1024,1024")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--prediction-type", choices=("eps", "x0"), default="x0")
    parser.add_argument("--reconstruction-loss", choices=("mse", "gears"), default="mse")
    parser.add_argument(
        "--autofocus-gamma",
        type=float,
        default=2.0,
        help="GEARS autofocus exponent offset; the error power is 2 + gamma.",
    )
    parser.add_argument(
        "--direction-lambda",
        type=float,
        default=1e-3,
        help="Weight for the GEARS direction-aware penalty.",
    )
    parser.add_argument(
        "--lambda-semantic-align",
        type=float,
        default=0.0,
        help="Weight for semantic alignment loss on z_basal (= z_sem - delta).",
    )
    parser.add_argument(
        "--lambda-direction-align",
        type=float,
        default=0.0,
        help="Weight for same-target guide direction alignment cosine loss.",
    )
    parser.add_argument("--guide-library-table", action="append", default=None, help="Mismatch library table(s) used to resolve each guide's fully matched sgRNA.")
    parser.add_argument("--activity-source", choices=("none", "predicted", "direct"), default="none")
    parser.add_argument("--guide-activity-cnn-table", default=None, help="Table_S8-style guide activity CNN input table.")
    parser.add_argument("--activity-table", default=None, help="Optional guide activity table for strength supervision.")
    parser.add_argument("--activity-col", default="relative_activity_day10")
    parser.add_argument("--guide-sequence-length", type=int, default=26)
    parser.add_argument("--guide-strength-hidden-dims", default="64,64")
    parser.add_argument("--guide-strength-dropout", type=float, default=0.0)
    parser.add_argument("--guide-strength-cnn-channels", default="32,32")
    parser.add_argument("--guide-strength-cnn-kernel-size", type=int, default=4)
    parser.add_argument("--guide-strength-cnn-dropout", type=float, default=0.25)
    parser.add_argument("--guide-strength-lambda", type=float, default=0.1)
    parser.add_argument(
        "--activity-condition-dim",
        type=int,
        default=16,
        help="Embedding dimension for the explicit activity condition passed into the DDIM denoiser.",
    )
    parser.add_argument(
        "--disable-explicit-activity-condition",
        action="store_true",
        help="Keep activity folded only into z_pert by disabling the explicit denoiser condition channel.",
    )
    parser.add_argument(
        "--enable-activity-delta-scaling",
        action="store_true",
        help="Also multiply perturbation delta by activity, matching the earlier folded-conditioning behavior.",
    )
    parser.add_argument("--signature-summary-json", help="Optional signature summary JSON for weighted reconstruction.")
    parser.add_argument("--signature-gene-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument(
        "--skip-delta-init",
        action="store_true",
        help="Skip scGen-style latent mean-difference initialization for perturbation embeddings.",
    )
    return parser.parse_args()


def load_dataset(
    data_path: str | Path,
    *,
    perturbation_col: str,
    activity_label_col: str | None,
    target_strategy: str,
    control_labels: tuple[str, ...],
    existing_mapping: dict[str, int] | None = None,
):
    return load_h5ad_perturbation_dataset(
        data_path,
        perturbation_col=perturbation_col,
        activity_label_col=activity_label_col,
        response_proxy_col=None,
        control_labels=control_labels,
        target_strategy=target_strategy,
        existing_mapping=existing_mapping,
    )


def attach_predicted_activity_features(
    dataset,
    *,
    guide_activity_cnn_table: str | Path,
    sequence_length: int,
    activity_lookup: dict[str, float] | None,
) -> dict[str, object]:
    records_by_key = load_guide_feature_records([guide_activity_cnn_table])
    if not records_by_key:
        raise ValueError("Guide activity CNN table was provided, but no records were loaded")
    labels = getattr(dataset, "activity_labels", None)
    if labels is None:
        raise ValueError("Dataset is missing activity_labels required for guide activity features")

    feature_dim = guide_cnn_feature_dim(sequence_length=sequence_length)
    feature_rows: list[np.ndarray] = []
    activity_targets: list[float] = []
    matched_feature_cells = 0
    missing_feature_cells = 0
    matched_activity_cells = 0
    missing_activity_cells = 0
    for pert_id, label in zip(dataset.perturbation_ids.tolist(), labels):
        if int(pert_id) == 0:
            feature_rows.append(np.zeros(feature_dim, dtype=np.float32))
            activity_targets.append(0.0)
            continue
        record = find_record_for_label(str(label), records_by_key)
        if record is None:
            feature_rows.append(np.zeros(feature_dim, dtype=np.float32))
            missing_feature_cells += 1
        else:
            feature_rows.append(
                build_notebook_cnn_feature_tensor(record, sequence_length=sequence_length).reshape(-1)
            )
            matched_feature_cells += 1
        if activity_lookup is None:
            activity_targets.append(0.0)
            continue
        value = None
        for key in canonical_guide_keys(str(label)):
            value = activity_lookup.get(key)
            if value is not None:
                break
        if value is None:
            activity_targets.append(0.0)
            missing_activity_cells += 1
        else:
            activity_targets.append(float(value))
            matched_activity_cells += 1

    dataset.guide_features = torch.from_numpy(np.stack(feature_rows, axis=0))
    dataset.guide_activity_targets = (
        torch.tensor(activity_targets, dtype=torch.float32)
        if activity_lookup is not None
        else None
    )
    return {
        "mode": "predicted",
        "feature_dim": feature_dim,
        "matched_feature_cells": matched_feature_cells,
        "missing_feature_cells": missing_feature_cells,
        "matched_activity_cells": matched_activity_cells,
        "missing_activity_cells": missing_activity_cells,
        "n_records": len(records_by_key),
        "activity_supervision": activity_lookup is not None,
    }


def lookup_activity_for_label(label: str, activity_lookup: dict[str, float]) -> float | None:
    for key in canonical_guide_keys(str(label)):
        value = activity_lookup.get(key)
        if value is not None:
            return float(value)
    return None


def attach_direct_activity_features(
    dataset,
    *,
    activity_lookup: dict[str, float],
) -> dict[str, object]:
    labels = getattr(dataset, "activity_labels", None)
    if labels is None:
        raise ValueError("Dataset is missing activity_labels required for direct activity features")

    feature_rows: list[np.ndarray] = []
    activity_targets: list[float] = []
    matched_activity_cells = 0
    missing_activity_cells = 0
    missing_labels: set[str] = set()
    for pert_id, label in zip(dataset.perturbation_ids.tolist(), labels):
        if int(pert_id) == 0:
            value = 0.0
        else:
            lookup_value = lookup_activity_for_label(str(label), activity_lookup)
            if lookup_value is None:
                value = 0.0
                missing_activity_cells += 1
                missing_labels.add(str(label))
            else:
                value = float(lookup_value)
                matched_activity_cells += 1
        feature_rows.append(np.asarray([value], dtype=np.float32))
        activity_targets.append(value)

    if missing_labels:
        preview = ", ".join(sorted(missing_labels)[:10])
        raise ValueError(
            f"Direct activity table is missing {len(missing_labels)} non-control labels: {preview}"
        )
    dataset.guide_features = torch.from_numpy(np.stack(feature_rows, axis=0))
    dataset.guide_activity_targets = torch.tensor(activity_targets, dtype=torch.float32)
    return {
        "mode": "direct",
        "feature_dim": 1,
        "matched_activity_cells": matched_activity_cells,
        "missing_activity_cells": missing_activity_cells,
        "activity_supervision": True,
    }


def prepare_activity_conditioning(
    datasets: list[object],
    *,
    activity_source: str,
    guide_activity_cnn_table: str | None,
    activity_table: str | None,
    activity_col: str,
    sequence_length: int,
) -> dict[str, object]:
    if activity_source == "none":
        return {"mode": "none", "feature_dim": 0, "activity_supervision": False}
    if activity_source == "direct":
        if activity_table is None:
            raise ValueError("Direct activity mode requires --activity-table")
        activity_lookup = load_guide_activity_lookup(activity_table, activity_col=activity_col)
        summary = None
        for dataset in datasets:
            current = attach_direct_activity_features(dataset, activity_lookup=activity_lookup)
            if summary is None:
                summary = current
        if summary is None:
            raise ValueError("No datasets were provided for activity conditioning")
        return {
            **summary,
            "activity_col": activity_col,
            "activity_table": activity_table,
            "guide_activity_cnn_table": None,
        }
    if not guide_activity_cnn_table:
        raise ValueError("Predicted activity mode requires --guide-activity-cnn-table")
    activity_lookup = (
        load_guide_activity_lookup(activity_table, activity_col=activity_col)
        if activity_table is not None
        else None
    )
    summary = None
    for dataset in datasets:
        current = attach_predicted_activity_features(
            dataset,
            guide_activity_cnn_table=guide_activity_cnn_table,
            sequence_length=sequence_length,
            activity_lookup=activity_lookup,
        )
        if summary is None:
            summary = current
    if summary is None:
        raise ValueError("No datasets were provided for activity conditioning")
    return {
        **summary,
        "activity_col": activity_col if activity_lookup is not None else None,
        "activity_table": activity_table,
        "guide_activity_cnn_table": guide_activity_cnn_table,
    }


@torch.no_grad()
def initialize_perturbation_embeddings(
    model: PerturbDiffusionModel,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, dict[int, int]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    zs: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []
    model.eval()
    for batch in loader:
        x0 = batch["x0"].to(device)
        perturbation_ids = batch["perturbation_ids"].to(device)
        z_sem = model.encoder(x0)
        zs.append(z_sem.detach().cpu())
        ids.append(perturbation_ids.detach().cpu())
    z_sem_all = torch.cat(zs, dim=0)
    perturbation_ids_all = torch.cat(ids, dim=0)
    # We initialize from the whole training set instead of one batch because we
    # want a stable estimate of each perturbation's average latent shift.
    return model.initialize_perturbation_from_latent_shift(
        z_sem_all.to(device),
        perturbation_ids_all.to(device),
    )


@torch.no_grad()
def compute_control_expression_mean(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total: torch.Tensor | None = None
    count = 0
    for batch in loader:
        x0 = batch["x0"].to(device)
        control_mask = batch["perturbation_ids"].to(device) == 0
        if not control_mask.any():
            continue
        control_x = x0[control_mask]
        batch_sum = control_x.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += int(control_x.shape[0])
    if total is None or count == 0:
        raise ValueError("control cells are required for GEARS direction-aware loss")
    return total / count


def evaluate_model(
    model: PerturbDiffusionModel,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running = {
        "loss": 0.0,
        "diffusion_loss": 0.0,
        "eval_mse_loss": 0.0,
        "semantic_align_loss": 0.0,
        "direction_align_loss": 0.0,
        "guide_strength_loss": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            out = model.training_losses(
                batch["x0"].to(device),
                batch["perturbation_ids"].to(device),
                batch["target_ids"].to(device),
                batch.get("guide_features", None).to(device) if "guide_features" in batch else None,
                batch.get("guide_activity_targets", None).to(device) if "guide_activity_targets" in batch else None,
            )
            for key in running:
                running[key] += float(out[key].detach().cpu())
            batches += 1
    if batches == 0:
        return {key: 0.0 for key in running}
    return {key: value / batches for key, value in running.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    control_labels = tuple(args.control_labels) if args.control_labels else CONTROL_PATTERNS

    if args.train_path and args.val_path and args.test_path:
        train_dataset, mapping = load_dataset(
            args.train_path,
            perturbation_col=args.perturbation_col,
            activity_label_col=args.activity_label_col,
            target_strategy=args.target_strategy,
            control_labels=control_labels,
        )
        val_dataset, _ = load_dataset(
            args.val_path,
            perturbation_col=args.perturbation_col,
            activity_label_col=args.activity_label_col,
            target_strategy=args.target_strategy,
            control_labels=control_labels,
            existing_mapping=mapping,
        )
        test_dataset, _ = load_dataset(
            args.test_path,
            perturbation_col=args.perturbation_col,
            activity_label_col=args.activity_label_col,
            target_strategy=args.target_strategy,
            control_labels=control_labels,
            existing_mapping=mapping,
        )
    else:
        try:
            import anndata as ad
        except ImportError as exc:
            raise ImportError("Automatic split needs anndata to read obs labels.") from exc

        full_dataset, mapping = load_dataset(
            args.data_path,
            perturbation_col=args.perturbation_col,
            activity_label_col=args.activity_label_col,
            target_strategy=args.target_strategy,
            control_labels=control_labels,
        )
        adata = ad.read_h5ad(args.data_path)
        labels = adata.obs[args.perturbation_col].astype(str).to_numpy()
        indices = list(range(len(labels)))
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_ratio,
            random_state=args.split_seed,
            stratify=labels,
        )
        train_val_labels = [labels[i] for i in train_val_idx]
        val_adjusted = args.val_ratio / (1.0 - args.test_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_adjusted,
            random_state=args.split_seed,
            stratify=train_val_labels,
        )
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        test_dataset = Subset(full_dataset, test_idx)

    base_datasets: list[object] = []
    for dataset in (train_dataset, val_dataset, test_dataset):
        base = dataset.dataset if isinstance(dataset, Subset) else dataset
        if all(base is not existing for existing in base_datasets):
            base_datasets.append(base)
    conditioning_summary = prepare_activity_conditioning(
        base_datasets,
        activity_source=args.activity_source,
        guide_activity_cnn_table=args.guide_activity_cnn_table,
        activity_table=args.activity_table,
        activity_col=args.activity_col,
        sequence_length=args.guide_sequence_length,
    )

    n_genes = train_dataset.dataset.n_genes if isinstance(train_dataset, Subset) else train_dataset.n_genes
    gene_names = (
        train_dataset.dataset.gene_names
        if isinstance(train_dataset, Subset) and train_dataset.dataset.gene_names is not None
        else (
            train_dataset.gene_names
            if hasattr(train_dataset, "gene_names") and train_dataset.gene_names is not None
            else tuple(str(i) for i in range(n_genes))
        )
    )
    cfg = ModelConfig(
        n_genes=n_genes,
        n_perturbations=len(mapping),
        latent_dim=args.latent_dim,
        encoder_hidden_dims=parse_hidden_dims(args.encoder_hidden_dims),
        denoiser_hidden_dims=parse_hidden_dims(args.denoiser_hidden_dims),
        diffusion=DiffusionConfig(
            timesteps=args.diffusion_steps,
            prediction_type=args.prediction_type,
        ),
        losses=TrainingLossConfig(
            signature_gene_weight=args.signature_gene_weight,
            reconstruction_loss=args.reconstruction_loss,
            autofocus_gamma=args.autofocus_gamma,
            direction_lambda=args.direction_lambda,
            lambda_semantic_align=args.lambda_semantic_align,
            lambda_direction_align=args.lambda_direction_align,
        ),
        guide_encoder=GuideFeatureEncoderConfig(
            enabled=args.activity_source in {"predicted", "direct"},
            feature_dim=int(conditioning_summary["feature_dim"]),
            activity_source=args.activity_source if args.activity_source != "none" else "predicted",
            explicit_condition=not args.disable_explicit_activity_condition,
            activity_condition_dim=args.activity_condition_dim,
            activity_scales_delta=args.enable_activity_delta_scaling,
            architecture="mlp" if args.activity_source == "direct" else "cnn",
            hidden_dims=parse_hidden_dims(args.guide_strength_hidden_dims),
            dropout=args.guide_strength_dropout,
            sequence_length=args.guide_sequence_length,
            cnn_channels=parse_hidden_dims(args.guide_strength_cnn_channels),
            cnn_kernel_size=args.guide_strength_cnn_kernel_size,
            cnn_dropout=args.guide_strength_cnn_dropout,
            lambda_strength=args.guide_strength_lambda if args.activity_source == "predicted" and args.activity_table else 0.0,
        ),
    )
    device = torch.device(args.device)
    model = PerturbDiffusionModel(cfg).to(device)
    if args.skip_delta_init:
        perturbation_init_counts = {"initialized": False}
    else:
        perturbation_init_counts = initialize_perturbation_embeddings(
            model,
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
    if args.activity_source in {"predicted", "direct"}:
        model.initialize_guide_strength_directions_from_perturbation_embeddings(
            torch.arange(0, len(mapping) + 1, dtype=torch.long, device=device)
        )
    signature_gene_weights = build_signature_gene_weights(
        gene_names,
        mapping,
        summary_path=args.signature_summary_json,
        signature_gene_weight=args.signature_gene_weight,
    )
    model.set_signature_gene_weights(signature_gene_weights.to(device))
    if args.reconstruction_loss == "gears":
        control_expression_mean = compute_control_expression_mean(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        model.set_control_expression_mean(control_expression_mean)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train/val/test loaders are kept separate because they answer different
    # questions: optimization, model selection, and final reporting.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with (output_dir / "perturbation_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, sort_keys=True)
    with (output_dir / "perturbation_init_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(perturbation_init_counts, handle, indent=2, sort_keys=True)

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_epoch = 0
    best_path = output_dir / "checkpoint_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            # training_losses returns both the scalar objective and extra
            # diagnostics, which is convenient for research experiments.
            out = model.training_losses(
                batch["x0"].to(device),
                batch["perturbation_ids"].to(device),
                batch["target_ids"].to(device),
                batch.get("guide_features", None).to(device) if "guide_features" in batch else None,
                batch.get("guide_activity_targets", None).to(device) if "guide_activity_targets" in batch else None,
            )
            out["loss"].backward()
            optimizer.step()
            running_loss += float(out["loss"].detach().cpu())
            batches += 1

        train_loss = running_loss / max(batches, 1)
        val_metrics = evaluate_model(model, val_loader, device=device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_diffusion_loss": val_metrics["diffusion_loss"],
            "val_eval_mse_loss": val_metrics["eval_mse_loss"],
            "val_semantic_align_loss": val_metrics["semantic_align_loss"],
            "val_direction_align_loss": val_metrics["direction_align_loss"],
            "val_guide_strength_loss": val_metrics["guide_strength_loss"],
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_eval_mse={val_metrics['eval_mse_loss']:.6f} "
            f"val_sem_align={val_metrics['semantic_align_loss']:.6f} "
            f"val_dir_align={val_metrics['direction_align_loss']:.6f} "
            f"val_strength={val_metrics['guide_strength_loss']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": cfg,
                    "perturbation_mapping": mapping,
                    "perturbation_init_counts": perturbation_init_counts,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                best_path,
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": cfg,
                    "perturbation_mapping": mapping,
                    "perturbation_init_counts": perturbation_init_counts,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                },
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
            )

    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    guide_scales = None
    guide_scale_rows = None
    if args.guide_library_table:
        guide_scales, guide_scale_rows = build_virtual_guide_scales_from_embedding(
            mapping,
            model.perturbation.embedding.weight.detach().cpu(),
            guide_library_tables=args.guide_library_table,
        )
        with (output_dir / "guide_virtual_scales.json").open("w", encoding="utf-8") as handle:
            json.dump(guide_scale_rows, handle, indent=2)

    test_metrics = evaluate_model(model, test_loader, device=device)
    print(
        f"best_epoch={best_epoch} best_val_loss={best_val:.6f} "
        f"test_loss={test_metrics['loss']:.6f}"
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
        "history": history,
        "perturbation_init_counts": perturbation_init_counts,
        "lambda_semantic_align": args.lambda_semantic_align,
        "lambda_direction_align": args.lambda_direction_align,
        "reconstruction_loss": args.reconstruction_loss,
        "autofocus_gamma": args.autofocus_gamma,
        "direction_lambda": args.direction_lambda,
        "guide_library_tables": args.guide_library_table,
        "virtual_s_enabled": guide_scales is not None,
        "conditioning_summary": conditioning_summary,
        "activity_source": args.activity_source,
        "activity_label_col": args.activity_label_col,
        "device": str(device),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
