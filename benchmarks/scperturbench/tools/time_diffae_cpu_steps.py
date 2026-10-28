"""Time a few real optimizer steps of the Diff-AE-Perturb training loop on this machine.

Used to decide whether the GSE132080 activity ablation can be trained on CPU in a tolerable
time, now that the RTX 5060 (sm_120) is known to be unusable with the image's torch builds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/Diff-AE-Perturb")
STEPS = int(sys.argv[2] if len(sys.argv) > 2 else 25)
sys.path.insert(0, str(REPO / "src"))

from perturbdiff.config import (  # noqa: E402
    DiffusionConfig,
    GuideFeatureEncoderConfig,
    ModelConfig,
    TrainingLossConfig,
)
from perturbdiff.model import PerturbDiffusionModel  # noqa: E402

N_GENES = 1019
N_PERTURBATIONS = 128
BATCH = 256
LATENT = 128
GUIDE_DIM = 4 * 26 * 2


def build(activity: bool) -> PerturbDiffusionModel:
    cfg = ModelConfig(
        n_genes=N_GENES,
        n_perturbations=N_PERTURBATIONS,
        latent_dim=LATENT,
        control_reference="per_cell",
        diffusion=DiffusionConfig(timesteps=1000, prediction_type="x0"),
        losses=TrainingLossConfig(),
        guide_encoder=GuideFeatureEncoderConfig(
            enabled=activity,
            feature_dim=GUIDE_DIM if activity else 0,
            activity_source="predicted",
            explicit_condition=True,
            activity_condition_dim=16,
            architecture="cnn",
        ),
    )
    return PerturbDiffusionModel(cfg)


def time_arm(activity: bool) -> dict[str, object]:
    torch.manual_seed(7)
    model = build(activity)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x0 = torch.rand(BATCH, N_GENES)
    perturbation_ids = torch.randint(1, N_PERTURBATIONS, (BATCH,))
    target_ids = perturbation_ids.clone()
    guide_features = torch.rand(BATCH, GUIDE_DIM) if activity else None

    model.train()
    for _ in range(3):  # warm-up
        out = model.training_losses(x0, perturbation_ids, target_ids, guide_features)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        optimizer.step()

    start = time.perf_counter()
    for _ in range(STEPS):
        out = model.training_losses(x0, perturbation_ids, target_ids, guide_features)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        optimizer.step()
    elapsed = time.perf_counter() - start

    steps_per_epoch = int(np.ceil(16285 / BATCH))
    per_step = elapsed / STEPS
    return {
        "activity": activity,
        "params": n_params,
        "seconds_per_step": round(per_step, 4),
        "steps_per_epoch": steps_per_epoch,
        "seconds_per_epoch": round(per_step * steps_per_epoch, 1),
        "minutes_for_200_epochs": round(per_step * steps_per_epoch * 200 / 60.0, 1),
        "torch_threads": torch.get_num_threads(),
    }


def main() -> int:
    print(json.dumps({"torch": torch.__version__, "device": "cpu"}, indent=2))
    results = [time_arm(False), time_arm(True)]
    for row in results:
        print(json.dumps(row, indent=2))
    total = sum(row["minutes_for_200_epochs"] for row in results)
    print(f"TOTAL minutes for both arms (200 epochs each): {total:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
