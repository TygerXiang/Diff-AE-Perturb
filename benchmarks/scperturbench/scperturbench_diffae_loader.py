"""Load a Diff-AE-Perturb checkpoint with the activity doser sized as it was at training time.

Why this exists
---------------
``PerturbDiffusionModel`` sizes two things from ``config.n_perturbations``:

  * the perturbation embedding table (guide-indexed, ``n_perturbations + 1`` rows), and
  * the target-specific activity doser (``GeneralizedSigmoid``, ``n_targets + 1`` tables).

Those are the same number only when the perturbation column is already the target.  For a guide-level
split they differ: the guide vocabulary is 103 while the target-gene vocabulary is 128.  A checkpoint
written by the fixed training path therefore carries a 104-row perturbation embedding *and* 129-wide
doser tables, while a default model construction produces 104 and 104.  Loading it with
``strict=False`` still raises ``size mismatch for guide_doser.beta`` (strict only tolerates
missing/unexpected keys), so the activity arm cannot be evaluated at all without this adjustment.

``n_perturbations`` must therefore stay at the guide vocabulary (the embedding is checkpoint-authoritative);
only the doser is rebuilt to the checkpoint's width, and the perturbation embedding is verified
afterwards to be sure nothing else drifted.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from perturbdiff.config import GuideFeatureEncoderConfig
from perturbdiff.model import PerturbDiffusionModel
from perturbdiff.modules import GeneralizedSigmoid


def doser_table_size(state_dict: dict) -> int | None:
    """Return the checkpoint's trained doser table width (``n_targets + 1``), if present."""
    bias = state_dict.get("guide_doser.bias")
    if bias is None or not hasattr(bias, "shape") or len(bias.shape) != 2:
        return None
    return int(bias.shape[1])


def activity_condition_present(state_dict: dict) -> bool:
    return any(key.startswith("activity_condition.") for key in state_dict)


def compatible_config_from_state(state: dict):
    """Upstream-compatible config reconstruction (no sizing changes)."""
    cfg = state["model_config"]
    state_dict = state["model_state_dict"]
    if not hasattr(cfg, "guide_encoder"):
        cfg = replace(cfg, guide_encoder=GuideFeatureEncoderConfig())
    if not hasattr(cfg, "control_reference"):
        # Legacy checkpoints never trained the counterfactual path; inference always fed one control
        # cell's latent per sample, so "per_cell" reproduces their historical numbers exactly.
        cfg = replace(cfg, control_reference="per_cell")

    has_activity_condition = activity_condition_present(state_dict)
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


def _resize_doser(model: PerturbDiffusionModel, width: int) -> None:
    current = model.guide_doser
    if current is None:
        raise RuntimeError("checkpoint has doser tables but the model has no guide doser")
    doser = GeneralizedSigmoid(
        n_targets=width - 1,
        non_linearity=getattr(current, "non_linearity", "sigm"),
    )
    model.guide_doser = doser.to(next(model.parameters()).device)


def load_model(path: Path, device: torch.device, *, report_missing: bool = True):
    """Return ``(model, perturbation_mapping, report)`` with the activity doser restored."""
    state = torch.load(Path(path), map_location=device, weights_only=False)
    state_dict = state["model_state_dict"]
    table_size = doser_table_size(state_dict)
    cfg = compatible_config_from_state(state)
    model = PerturbDiffusionModel(cfg).to(device)

    resized = False
    if table_size is not None and model.guide_doser is not None:
        if int(model.guide_doser.bias.shape[1]) != table_size:
            # Widen only the doser; the perturbation embedding stays checkpoint-authoritative.
            _resize_doser(model, table_size)
            resized = True

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical_missing = [key for key in missing if key.startswith("guide_doser.")]

    embedding = model.perturbation.embedding.weight
    report = {
        "checkpoint": str(path),
        "doser_table_size_in_checkpoint": table_size,
        "doser_table_size_in_model": (
            None if model.guide_doser is None else int(model.guide_doser.bias.shape[1])
        ),
        "doser_resized_at_load": resized,
        "embedding_rows_in_model": int(embedding.shape[0]),
        "embedding_rows_in_checkpoint": int(state_dict["perturbation.embedding.weight"].shape[0]),
        "n_perturbations_in_config": int(cfg.n_perturbations),
        "n_perturbations_in_checkpoint_mapping": len(state["perturbation_mapping"]),
        "n_missing_keys": len(missing),
        "n_unexpected_keys": len(unexpected),
        "missing_doser_keys": critical_missing,
        "missing_keys": [] if not report_missing else list(missing),
    }
    if critical_missing:
        raise RuntimeError(
            "refusing to evaluate: the trained activity doser was not restored "
            f"({critical_missing}); the activity arm would run with a neutral doser"
        )
    if report["embedding_rows_in_model"] != report["embedding_rows_in_checkpoint"]:
        raise RuntimeError(
            "perturbation embedding size drifted: model "
            f"{report['embedding_rows_in_model']} vs checkpoint "
            f"{report['embedding_rows_in_checkpoint']}"
        )
    model.eval()
    mapping = {str(key): int(value) for key, value in state["perturbation_mapping"].items()}
    return model, mapping, report
