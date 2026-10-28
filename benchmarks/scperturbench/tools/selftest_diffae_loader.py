#!/usr/bin/env python3
"""Verify the checkpoint loader restores a trained activity doser (synthetic checkpoint, no training).

Builds a guide-level-style checkpoint in memory where the target vocabulary (128) exceeds the guide
vocabulary (103), saves it, then checks that:

  * the upstream-style load (doser sized from ``config.n_perturbations``) silently drops the trained
    doser tables, and
  * ``scperturbench_diffae_loader.load_model`` restores them and would refuse to proceed if it could not.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

import sys

WORK = Path("/root/perturbench")
REPO = Path("/root/Diff-AE-Perturb")
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "tools"))
sys.path.insert(0, str(REPO / "src"))

import scperturbench_diffae_loader as loader  # noqa: E402
from perturbdiff.config import (  # noqa: E402
    DiffusionConfig,
    GuideFeatureEncoderConfig,
    ModelConfig,
    TrainingLossConfig,
)
from perturbdiff.model import PerturbDiffusionModel  # noqa: E402

GUIDE_VOCAB = 103
TARGET_VOCAB = 128
N_GENES = 16
GUIDE_FEATURES = 4 * 26 * 2


def build_config(n_perturbations: int) -> ModelConfig:
    return ModelConfig(
        n_genes=N_GENES,
        n_perturbations=n_perturbations,
        latent_dim=8,
        control_reference="per_cell",
        diffusion=DiffusionConfig(timesteps=50, prediction_type="x0"),
        losses=TrainingLossConfig(),
        guide_encoder=GuideFeatureEncoderConfig(
            enabled=True,
            feature_dim=GUIDE_FEATURES,
            activity_source="predicted",
            explicit_condition=True,
            activity_condition_dim=4,
            architecture="cnn",
            hidden_dims=(8, 8),
            cnn_channels=(4,),
            lambda_strength=0.1,
        ),
    )


def build_trained_state() -> dict:
    """A checkpoint as the fixed training run writes it: config keeps the guide vocab, doser is wider."""
    cfg = build_config(GUIDE_VOCAB)
    model = PerturbDiffusionModel(cfg)
    # Simulate a trained doser: the real run leaves it near beta=1/bias=0 but with target-specific rows.
    with torch.no_grad():
        model.guide_doser.beta.normal_(1.0, 0.05)
        model.guide_doser.bias.normal_(0.0, 0.05)

    state_dict = model.state_dict()
    # Emulate what the model would write if built with the corrected targets: replace the doser tables.
    grown = PerturbDiffusionModel(build_config(TARGET_VOCAB))
    with torch.no_grad():
        grown.guide_doser.beta.normal_(1.0, 0.05)
        grown.guide_doser.bias.normal_(0.0, 0.05)
    state_dict["guide_doser.beta"] = grown.guide_doser.beta.detach().clone()
    state_dict["guide_doser.bias"] = grown.guide_doser.bias.detach().clone()

    mapping = {"control": 0}
    mapping.update({f"GENE{i}_GENE{i}_+_1.23-P1P2_0{i % 10}": i for i in range(1, GUIDE_VOCAB)})
    return {
        "model_state_dict": state_dict,
        "model_config": cfg,
        "perturbation_mapping": mapping,
        "epoch": 200,
        "val_loss": 0.05,
    }


def main() -> int:
    state = build_trained_state()
    size = loader.doser_table_size(state["model_state_dict"])
    assert size == TARGET_VOCAB + 1, size
    print(f"[ok] checkpoint carries doser table of width {size}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.pt"
        torch.save(state, path)
        device = torch.device("cpu")

        # --- upstream-style load cannot restore the doser --------------------------------------
        # Note: strict=False only tolerates missing/unexpected *keys*; a size mismatch on an existing
        # key still raises.  So the upstream path fails loudly here rather than degrading silently --
        # which is exactly why the audited loader must be what the harness uses.
        cfg_default = loader.compatible_config_from_state(
            {**state, "model_state_dict": {k: v for k, v in state["model_state_dict"].items()
                                           if not k.startswith("guide_doser.")}}
        )
        naive = PerturbDiffusionModel(cfg_default)
        assert naive.guide_doser.bias.shape == (1, GUIDE_VOCAB + 1), naive.guide_doser.bias.shape
        try:
            naive.load_state_dict(state["model_state_dict"], strict=False)
        except RuntimeError as exc:
            message = str(exc)
            assert "guide_doser.beta" in message and "size mismatch" in message, message
            print("[expected] upstream-style load fails on doser size mismatch (not silent)")
        else:
            dropped = [k for k in naive.state_dict() if k.startswith("guide_doser.")]
            raise AssertionError(f"expected a doser size mismatch; got a silent load of {dropped}")

        # --- audited loader restores them ----------------------------------------------------
        model, mapping, report = loader.load_model(path, device)
        assert report["doser_table_size_in_model"] == TARGET_VOCAB + 1, report
        assert not report["missing_doser_keys"], report
        trained_bias = state["model_state_dict"]["guide_doser.bias"]
        assert torch.allclose(model.guide_doser.bias, trained_bias), "doser bias not restored"
        print(f"[ok] audited load restored doser (model width {report['doser_table_size_in_model']})")

        # --- a genuinely unrestorable doser must raise --------------------------------------
        broken = {**state}
        broken["model_state_dict"] = dict(state["model_state_dict"])
        broken["model_state_dict"]["guide_doser.bias"] = torch.zeros(1, 4)
        broken_path = Path(tmp) / "broken.pt"
        torch.save(broken, broken_path)
        try:
            loader.load_model(broken_path, device)
        except RuntimeError as exc:
            message = str(exc)
            assert "guide_doser" in message, message
            print("[ok] refuses a checkpoint whose doser cannot be restored")
        else:
            print("UNEXPECTED: loader accepted an unrestorable doser")
            return 1

    print("SELF-TEST PASS: checkpoint loader restores the trained activity doser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
