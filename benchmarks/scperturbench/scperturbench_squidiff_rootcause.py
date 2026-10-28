"""Root-cause diagnostic for the Squidiff sampling blow-up on GSE132080.

The previous benchmark run trained Squidiff to a healthy eps-prediction loss but sampled
cells whose scale was ~40x the training data.  This script separates the two candidate
mechanisms using the *existing* checkpoint (no retraining):

1. ``eps`` shrinkage at high noise levels -- if the model under-predicts the noise, DDIM
   grows the sample by ``exp(0.5 * sum_t beta_t * (1 - gamma_t))`` where ``gamma_t`` is the
   measured regression coefficient of the prediction onto the true noise.
2. ``process_xstart`` rectification -- Squidiff's ``diffusion.py`` keeps guided-diffusion's
   ``clip_denoised`` flag but, when it is False, still calls ``x.clamp(0,)``.  ``pred_xstart``
   is ill-conditioned at high t (``1/sqrt(alpha_cumprod)`` up to ~190), so clamping a
   zero-mean, huge-scale quantity at zero injects a large positive drift at every step.

Everything here runs Squidiff's own code; the only intervention is the ``process_xstart``
policy, which is switched by monkeypatching a faithful copy onto the diffusion instance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch as th


def build_sampler(repo: str, checkpoint: Path, gene_size: int):
    if Path(repo).exists():
        sys.path.insert(0, repo)
    import sample_squidiff

    return sample_squidiff.sampler(
        model_path=str(checkpoint),
        gene_size=gene_size,
        output_dim=gene_size,
        use_drug_structure=False,
    )


from scperturbench_squidiff_sampling import POLICIES, sample_progressive

def eps_diagnostics(model, diffusion, z_sem, cells, seed: int, device) -> dict:
    """Measure the model's eps prediction quality as a function of t."""
    gen = th.Generator(device="cpu").manual_seed(seed)
    x0_all = th.as_tensor(cells, dtype=th.float32)
    buckets = [(0, 100), (100, 250), (250, 500), (500, 750), (750, 1000)]
    out: dict = {"buckets": []}
    for low, high in buckets:
        n = 256
        index = th.randint(0, z_sem.shape[0], (n,), generator=gen)
        x0 = x0_all[index].to(device)
        t = th.randint(low, max(low + 1, high), (n,), generator=gen).to(device)
        noise = th.randn(x0.shape, generator=gen).to(device)
        x_t = diffusion.q_sample(x0, t, noise=noise)
        eps_pred = model(x_t, diffusion._scale_timesteps(t), z_mod=z_sem[index.to(device)])
        eps_true = noise
        num = float((eps_pred * eps_true).sum())
        den = float((eps_true * eps_true).sum())
        gamma = num / max(den, 1e-12)
        mse = float(((eps_pred - eps_true) ** 2).mean())
        pred_norm = float(eps_pred.norm(dim=1).mean())
        true_norm = float(eps_true.norm(dim=1).mean())
        a = float(diffusion.alphas_cumprod[low])
        pred_flat = eps_pred.detach().flatten().float().cpu().numpy()
        true_flat = eps_true.detach().flatten().float().cpu().numpy()
        out["buckets"].append(
            {
                "t_low": low,
                "t_high": high,
                "mse_vs_true_noise": mse,
                "gamma_projection": gamma,
                "corr": float(np.corrcoef(pred_flat, true_flat)[0, 1]),
                "pred_norm_over_true_norm": pred_norm / max(true_norm, 1e-12),
                "sqrt_alpha_cumprod_at_t_low": float(np.sqrt(a)),
                "amplification_1_over_sqrt_a": float(1.0 / np.sqrt(a)),
            }
        )
    # Analytic DDIM growth factor implied by the measured shrinkage.
    betas = diffusion.betas
    gamma_by_t = np.ones_like(betas)
    for bucket in out["buckets"]:
        gamma_by_t[bucket["t_low"] : bucket["t_high"]] = bucket["gamma_projection"]
    growth = float(np.exp(0.5 * np.sum(betas * (1.0 - gamma_by_t))))
    out["implied_growth_factor_exp_half_sum_beta_1_minus_gamma"] = growth
    return out


def trace_sampling(model, diffusion, z_sem, args, mode: str, gene_size: int):
    """Run Squidiff's own DDIM loop while recording per-step sample scale."""
    trace = []
    final = None
    local, gen_loop = sample_progressive(
        types.SimpleNamespace(model=model, diffusion=diffusion), z_sem, gene_size, mode
    )
    for step, sample in enumerate(gen_loop):
        x = sample["sample"]
        px0 = sample.get("pred_xstart")
        t_value = int(local.timestep_map[len(local.timestep_map) - 1 - step])
        trace.append(
            {
                "step": step,
                "t": t_value,
                "x_mean": float(x.mean()),
                "x_max": float(x.max()),
                "x_absmean": float(x.abs().mean()),
                "x_norm": float(x.norm(dim=1).mean()),
                "pred_x0_mean": float(px0.mean()) if px0 is not None else float("nan"),
                "pred_x0_max": float(px0.max()) if px0 is not None else float("nan"),
                "pred_x0_absmean": float(px0.abs().mean()) if px0 is not None else float("nan"),
            }
        )
        final = sample
    return final["sample"], trace


def summarize(name: str, cells: np.ndarray) -> dict:
    flat = np.asarray(cells, dtype=np.float64)
    return {
        "variant": name,
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "max": float(flat.max()),
        "min": float(flat.min()),
        "p99": float(np.percentile(flat, 99)),
        "frac_zero": float((flat == 0).mean()),
        "pseudobulk_mean": float(flat.mean(axis=0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--group-map", type=Path, required=True)
    parser.add_argument("--condition", default="ALDOA")
    parser.add_argument("--n-cells", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--modes",
        default="upstream,none,neg1_1",
        help=f"comma-separated process_xstart policies from {POLICIES}",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--repo", default="/root/Squidiff")
    parser.add_argument("--skip-trace", action="store_true")
    args = parser.parse_args()

    import anndata as ad

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = ad.read_h5ad(args.train_h5ad)
    group_map = json.loads(Path(args.group_map).read_text())
    gene_size = train.n_vars
    sampler = build_sampler(args.repo, args.checkpoint, gene_size)
    model, diffusion = sampler.model, sampler.diffusion
    report: dict = {
        "checkpoint": str(args.checkpoint),
        "condition": args.condition,
        "gene_size": int(gene_size),
        "diffusion_steps": int(diffusion.num_timesteps),
        "timestep_respacing": "official default (all timesteps)",
        "seed": args.seed,
    }

    group = group_map[args.condition]
    mask = np.asarray(train.obs["Group"]) == group
    cells = np.asarray(train.X[mask].todense() if hasattr(train.X[mask], "todense") else train.X[mask], dtype=np.float32)
    report["n_condition_cells"] = int(cells.shape[0])
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(cells.shape[0], min(args.n_cells, cells.shape[0]), replace=False)
    x_start = th.from_numpy(cells[pick])
    device = next(model.parameters()).device
    x_start = x_start.to(device)
    with th.no_grad():
        z_sem = model.encoder(x_start, label=None, drug_dose=None)

    report["real_data_stats"] = summarize("training cells (this condition)", cells)

    with th.no_grad():
        report["eps_diagnostics"] = eps_diagnostics(model, diffusion, z_sem, cells, args.seed, device)

    variants = {}
    traces = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        started = time.time()
        with th.no_grad():
            generated, trace = trace_sampling(model, diffusion, z_sem, args, mode, int(gene_size))
        generated_np = generated.detach().cpu().numpy().astype(np.float32)
        variants[mode] = summarize(f"mode={mode}", generated_np)
        variants[mode]["seconds"] = round(time.time() - started, 1)
        if not args.skip_trace:
            keep = [0, 1, 2, 5, 10, 50, 100, 250, 500, 750, 900, 990, len(trace) - 1]
            traces[mode] = [trace[i] for i in sorted(set(i for i in keep if 0 <= i < len(trace)))]
    report["generated_stats"] = variants
    report["progress"] = traces

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "progress"}, indent=2))
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
