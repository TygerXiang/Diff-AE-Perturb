from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import torch
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.model import PerturbDiffusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a trained PerturbDiff checkpoint generates expression-scale samples."
    )
    parser.add_argument(
        "--adata-path",
        type=Path,
        default=REPO_ROOT / "reproducibility" / "prepared" / "splits" / "gse132080_test.h5ad",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs" / "gse132080_with_split_1000ep" / "checkpoint_best.pt",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "runs" / "generation_sanity" / "perturbdiff_generation_sanity.json",
    )
    parser.add_argument("--perturbation-col", default="perturbation_label")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--target", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def tensor_stats(x: torch.Tensor | np.ndarray, *, real_p99: float | None = None) -> dict[str, float]:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().float().numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
    flat = arr.reshape(-1)
    out = {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "frac_negative": float(np.mean(flat < 0)),
    }
    if real_p99 is not None:
        out["frac_above_real_p99"] = float(np.mean(flat > real_p99))
    return out


def label_ids(labels: np.ndarray, mapping: dict[str, int], control_label: str) -> torch.Tensor:
    ids: list[int] = []
    for label in labels:
        value = str(label)
        if value == control_label:
            ids.append(0)
        else:
            ids.append(int(mapping[value]))
    return torch.tensor(ids, dtype=torch.long)


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


@torch.no_grad()
def partial_ddim_from_noised_real(
    model: PerturbDiffusionModel,
    x_start: torch.Tensor,
    start_timestep: int,
    z_cond: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    if start_timestep < 1:
        return x_start
    device = x_start.device
    batch_size = x_start.shape[0]
    t_start = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
    x, _ = model.diffusion.q_sample(x_start, t_start)
    time_seq = torch.linspace(start_timestep, 0, steps + 1, device=device).round().long()
    time_seq = torch.unique_consecutive(time_seq)
    if time_seq[-1].item() != 0:
        time_seq = torch.cat([time_seq, torch.zeros(1, device=device, dtype=torch.long)])

    for idx in range(len(time_seq) - 1):
        t_value = int(time_seq[idx].item())
        prev_t_value = int(time_seq[idx + 1].item())
        t = torch.full((batch_size,), t_value, device=device, dtype=torch.long)
        prev_t = torch.full((batch_size,), prev_t_value, device=device, dtype=torch.long)
        prediction = model.denoiser(x, t, z_cond)
        x0, eps = model.diffusion.prediction_to_x0_and_eps(x, t, prediction)
        prev_alpha_bar = extract(model.diffusion.alphas_cumprod, prev_t, x.shape)
        x = torch.sqrt(prev_alpha_bar) * x0 + torch.sqrt(1.0 - prev_alpha_bar) * eps
    return x


@torch.no_grad()
def one_step_x0_metrics(
    model: PerturbDiffusionModel,
    x0: torch.Tensor,
    perturbation_ids: torch.Tensor,
    timesteps: list[int],
    *,
    real_p99: float,
) -> dict[str, dict[str, float | dict[str, float]]]:
    z_cond = model.encode_condition(x0, perturbation_ids)["z_pert"]
    out: dict[str, dict[str, float | dict[str, float]]] = {}
    for timestep in timesteps:
        t = torch.full((x0.shape[0],), timestep, device=x0.device, dtype=torch.long)
        x_t, noise = model.diffusion.q_sample(x0, t)
        prediction = model.denoiser(x_t, t, z_cond)
        x0_hat, eps_pred = model.diffusion.prediction_to_x0_and_eps(x_t, t, prediction)
        prediction_target = x0 if model.diffusion.prediction_type == "x0" else noise
        out[str(timestep)] = {
            "prediction_mse": float(torch.mean((prediction - prediction_target).square()).detach().cpu()),
            "eps_mse": float(torch.mean((eps_pred - noise).square()).detach().cpu()),
            "x0_mse": float(torch.mean((x0_hat - x0).square()).detach().cpu()),
            "x0_hat_stats": tensor_stats(x0_hat, real_p99=real_p99),
        }
    return out


def choose_target(labels: np.ndarray, mapping: dict[str, int], control_label: str, requested: str | None) -> str:
    if requested is not None:
        return requested
    values, counts = np.unique(labels, return_counts=True)
    ranked = sorted(zip(values, counts), key=lambda item: int(item[1]), reverse=True)
    for value, _ in ranked:
        value = str(value)
        if value != control_label and value in mapping:
            return value
    raise ValueError("Could not choose a non-control target present in checkpoint mapping")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available in this Python environment; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    adata = ad.read_h5ad(args.adata_path)
    labels = adata.obs[args.perturbation_col].astype(str).to_numpy()
    x_np = to_dense(adata.X).astype(np.float32, copy=False)
    real_stats = tensor_stats(x_np)
    real_p99 = real_stats["p99"]

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PerturbDiffusionModel(state["model_config"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    mapping = {str(key): int(value) for key, value in state["perturbation_mapping"].items()}

    usable_mask = (labels == args.control_label) | np.isin(labels, list(mapping.keys()))
    usable_indices = np.flatnonzero(usable_mask)
    if usable_indices.size == 0:
        raise ValueError("No usable cells found for checkpoint mapping")
    rng = np.random.default_rng(args.seed)
    batch_indices = rng.choice(
        usable_indices,
        size=min(args.batch_size, usable_indices.size),
        replace=False,
    )
    batch_x = torch.from_numpy(x_np[batch_indices]).to(device)
    batch_labels = labels[batch_indices]
    batch_ids = label_ids(batch_labels, mapping, args.control_label).to(device)

    control_indices = np.flatnonzero(labels == args.control_label)
    if control_indices.size == 0:
        raise ValueError(f"No control cells with label {args.control_label!r}")
    control_indices = rng.choice(
        control_indices,
        size=min(args.batch_size, control_indices.size),
        replace=False,
    )
    control_x = torch.from_numpy(x_np[control_indices]).to(device)
    control_ids = torch.zeros(control_x.shape[0], dtype=torch.long, device=device)
    control_z = model.encode_condition(control_x, control_ids)["z_pert"]

    target = choose_target(labels, mapping, args.control_label, args.target)
    target_id = mapping[target]
    target_ids = torch.full((control_x.shape[0],), target_id, dtype=torch.long, device=device)

    one_step_timesteps = [1, 10, 50, 100, 250, 500, model.diffusion.timesteps]
    partial_starts = [10, 50, 100, 250, 500, model.diffusion.timesteps]

    with torch.no_grad():
        control_ddim = model.diffusion.ddim_sample(
            model.denoiser,
            shape=tuple(control_x.shape),
            z_cond=control_z,
            steps=args.ddim_steps,
        )
        target_ddim = model.sample_counterfactual(
            control_x,
            target_ids,
            steps=args.ddim_steps,
        )
        batch_z = model.encode_condition(batch_x, batch_ids)["z_pert"]
        partial = {}
        for t in partial_starts:
            sample = partial_ddim_from_noised_real(
                model,
                batch_x,
                min(t, model.diffusion.timesteps),
                batch_z,
                steps=min(args.ddim_steps, max(1, t)),
            )
            partial[str(t)] = {
                "mse_to_real": float(torch.mean((sample - batch_x).square()).detach().cpu()),
                "stats": tensor_stats(sample, real_p99=real_p99),
            }

    out = {
        "config": {
            "adata_path": str(args.adata_path),
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "batch_size": int(batch_x.shape[0]),
            "ddim_steps": args.ddim_steps,
            "prediction_type": model.diffusion.prediction_type,
            "target": target,
        },
        "real_expression_stats": real_stats,
        "one_step_x0_reconstruction": one_step_x0_metrics(
            model,
            batch_x,
            batch_ids,
            one_step_timesteps,
            real_p99=real_p99,
        ),
        "ddim_from_pure_noise": {
            "control_condition": tensor_stats(control_ddim, real_p99=real_p99),
            "target_condition": tensor_stats(target_ddim, real_p99=real_p99),
        },
        "partial_ddim_from_noised_real": partial,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved generation sanity check to {args.output_json}")
    print(json.dumps({
        "real": real_stats,
        "ddim_from_pure_noise": out["ddim_from_pure_noise"],
    }, indent=2))


if __name__ == "__main__":
    main()
