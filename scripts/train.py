from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff import DiffusionConfig, ModelConfig, PerturbDiffusionModel, TrainingLossConfig
from perturbdiff.data import (
    CONTROL_PATTERNS,
    build_signature_gene_weights,
    create_dataloader,
    load_10x_perturbation_dataset,
    load_h5ad_perturbation_dataset,
)


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PerturbDiff on real single-cell perturbation data.")
    parser.add_argument("--data-format", choices=("10x", "h5ad"), required=True)
    parser.add_argument("--data-path", required=True, help="10x directory or .h5ad path.")
    parser.add_argument("--metadata-path", help="CSV/TSV metadata path for 10x data.")
    parser.add_argument("--perturbation-col", required=True, help="Metadata/obs column containing guide or target labels.")
    parser.add_argument("--barcode-col", default="cell_barcode")
    parser.add_argument("--control-label", action="append", dest="control_labels")
    parser.add_argument("--target-strategy", choices=("first_token", "raw"), default="first_token")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--encoder-hidden-dims", default="512,256")
    parser.add_argument("--denoiser-hidden-dims", default="1024,1024")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--prediction-type", choices=("eps", "x0"), default="eps")
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
    parser.add_argument("--signature-summary-json", help="Optional response-proxy summary JSON for signature weighting.")
    parser.add_argument("--signature-gene-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--skip-delta-init",
        action="store_true",
        help="Skip scGen-style latent mean-difference initialization for perturbation embeddings.",
    )
    return parser.parse_args()


def load_dataset(args: argparse.Namespace):
    control_labels = tuple(args.control_labels) if args.control_labels else CONTROL_PATTERNS
    if args.data_format == "h5ad":
        return load_h5ad_perturbation_dataset(
            args.data_path,
            perturbation_col=args.perturbation_col,
            control_labels=control_labels,
            target_strategy=args.target_strategy,
        )
    if not args.metadata_path:
        raise ValueError("--metadata-path is required for --data-format 10x")
    return load_10x_perturbation_dataset(
        args.data_path,
        metadata_path=args.metadata_path,
        perturbation_col=args.perturbation_col,
        barcode_col=args.barcode_col,
        control_labels=control_labels,
        target_strategy=args.target_strategy,
    )


@torch.no_grad()
def initialize_perturbation_embeddings(
    model: PerturbDiffusionModel,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, dict[int, int] | dict[int, float]]:
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, perturbation_mapping = load_dataset(args)
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    cfg = ModelConfig(
        n_genes=dataset.n_genes,
        n_perturbations=len(perturbation_mapping),
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
        ),
    )
    device = torch.device(args.device)
    model = PerturbDiffusionModel(cfg).to(device)
    if args.skip_delta_init:
        perturbation_init_counts = {"initialized": False}
    else:
        perturbation_init_counts = initialize_perturbation_embeddings(
            model,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
    signature_gene_weights = build_signature_gene_weights(
        dataset.gene_names if dataset.gene_names is not None else tuple(str(i) for i in range(dataset.n_genes)),
        perturbation_mapping,
        summary_path=args.signature_summary_json,
        signature_gene_weight=args.signature_gene_weight,
    )
    model.set_signature_gene_weights(signature_gene_weights.to(device))
    if args.reconstruction_loss == "gears":
        control_expression_mean = compute_control_expression_mean(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        model.set_control_expression_mean(control_expression_mean)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with (output_dir / "perturbation_mapping.json").open("w") as handle:
        json.dump(perturbation_mapping, handle, indent=2, sort_keys=True)
    with (output_dir / "perturbation_init_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(perturbation_init_counts, handle, indent=2, sort_keys=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in dataloader:
            x0 = batch["x0"].to(device)
            perturbation_ids = batch["perturbation_ids"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model.training_losses(
                x0,
                perturbation_ids,
                batch["target_ids"].to(device),
            )
            out["loss"].backward()
            optimizer.step()

            running_loss += float(out["loss"].detach().cpu())
            batches += 1

        mean_loss = running_loss / max(batches, 1)
        print(f"epoch={epoch} loss={mean_loss:.6f}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": cfg,
                    "perturbation_mapping": perturbation_mapping,
                    "perturbation_init_counts": perturbation_init_counts,
                    "epoch": epoch,
                    "loss": mean_loss,
                },
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
            )


if __name__ == "__main__":
    main()
