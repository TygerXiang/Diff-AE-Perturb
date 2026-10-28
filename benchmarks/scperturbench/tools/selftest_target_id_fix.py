#!/usr/bin/env python3
"""Self-test for the activity-path target-id sizing fix (no dataset required).

Reproduces the exact failure mode: a guide-level split where the guide vocabulary is smaller than
the target-gene vocabulary, then checks that
  (a) the unpatched model raises IndexError on a high target id, and
  (b) the patched model builds and runs a full training_losses step with those ids.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

WORK = Path("/root/perturbench")
sys.path.insert(0, str(WORK / "tools"))
sys.path.insert(0, "/root/Diff-AE-Perturb/src")

from perturbdiff.config import (  # noqa: E402
    DiffusionConfig,
    GuideFeatureEncoderConfig,
    ModelConfig,
    TrainingLossConfig,
)
from perturbdiff.model import PerturbDiffusionModel  # noqa: E402

GUIDE_VOCAB = 103     # guides + control (the perturbation vocabulary)
TARGET_VOCAB = 128    # target genes + control (the doser vocabulary)
BATCH = 8
N_GENES = 32
GUIDE_FEATURES = 4 * 26 * 2


def build_config() -> ModelConfig:
    return ModelConfig(
        n_genes=N_GENES,
        n_perturbations=GUIDE_VOCAB,
        latent_dim=16,
        control_reference="per_cell",
        diffusion=DiffusionConfig(timesteps=100, prediction_type="x0"),
        losses=TrainingLossConfig(),
        guide_encoder=GuideFeatureEncoderConfig(
            enabled=True,
            feature_dim=GUIDE_FEATURES,
            activity_source="predicted",
            explicit_condition=True,
            activity_condition_dim=8,
            architecture="cnn",
            hidden_dims=(16, 16),
            cnn_channels=(4,),
            lambda_strength=0.1,
        ),
    )


def make_batch():
    torch.manual_seed(0)
    x0 = torch.rand(BATCH, N_GENES)
    # Guide ids stay inside the perturbation vocabulary ...
    perturbation_ids = torch.randint(1, GUIDE_VOCAB, (BATCH,))
    # ... while target ids use the (larger) target vocabulary, exactly as the data loader produces.
    target_ids = torch.tensor([1, 4, 26, 60, 100, 120, 126, 127])
    guide_features = torch.rand(BATCH, GUIDE_FEATURES)
    return x0, perturbation_ids, target_ids, guide_features


def main() -> int:
    import patch_perturbdiff_target_ids as shim
    from perturbdiff.model import PerturbDiffusionModel as Model

    assert shim.PATCH_APPLIED, "shim should install itself on import"
    original_init = Model._original_init
    patched_init = Model.__init__

    x0, perturbation_ids, target_ids, guide_features = make_batch()
    assert int(target_ids.max()) >= GUIDE_VOCAB, "test must exercise an out-of-range target id"

    # --- unpatched behaviour: must fail -------------------------------------------------------
    Model.__init__ = original_init
    try:
        model = Model(build_config())
        model.training_losses(x0, perturbation_ids, target_ids, guide_features)
        print("UNEXPECTED: unpatched model accepted out-of-range target ids")
        return 1
    except IndexError as exc:
        print(f"[expected] unpatched model fails: {type(exc).__name__}: {exc}")
    finally:
        Model.__init__ = patched_init

    # --- patched behaviour: must work ---------------------------------------------------------
    shim.set_target_vocab_size(TARGET_VOCAB)
    model = Model(build_config())
    doser = model.guide_doser
    assert doser is not None, "patch removed the doser"
    assert doser.bias.shape == (1, TARGET_VOCAB + 1), doser.bias.shape
    assert doser.beta.shape == (1, TARGET_VOCAB + 1), doser.beta.shape
    out = model.training_losses(x0, perturbation_ids, target_ids, guide_features)
    loss = float(out["loss"])
    assert torch.isfinite(out["loss"]), loss
    print(f"[ok] patched model steps fine (loss={loss:.4f}); doser tables={tuple(doser.bias.shape)}")

    # gradient must reach the newly sized doser rows used by the high target ids
    out["loss"].backward()
    grad = doser.bias.grad
    assert grad is not None and torch.isfinite(grad).all()
    touched = (grad.abs().sum(dim=0) > 0).nonzero().flatten().tolist()
    assert max(touched) >= GUIDE_VOCAB, f"high target-id rows received no gradient: {touched}"
    print(f"[ok] doser rows receiving gradient include high target ids up to {max(touched)}")

    # --- doser must not have been resized when it is unnecessary -------------------------------
    shim.set_target_vocab_size(GUIDE_VOCAB)
    model_small = Model(build_config())
    assert model_small.guide_doser.bias.shape == (1, GUIDE_VOCAB + 1), model_small.guide_doser.bias.shape
    print("[ok] no resize when the target vocabulary already fits")

    print("SELF-TEST PASS: activity-path target-id sizing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
