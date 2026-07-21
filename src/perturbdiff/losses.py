from __future__ import annotations

import torch


def weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    error = (prediction - target).square()
    if weights is None:
        return error.mean()
    if weights.shape != error.shape:
        raise ValueError("weights must match prediction/target shape")
    # Normalize by the total weight of each sample so larger raw weight sums do
    # not automatically dominate the batch loss.
    per_sample_weight = weights.sum(dim=-1).clamp_min(eps)
    per_sample = (error * weights).sum(dim=-1).div(per_sample_weight)
    return per_sample.mean()


def gears_autofocus_direction_loss(
    prediction_x0: torch.Tensor,
    target_x0: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    control_mean: torch.Tensor | None = None,
    gamma: float = 2.0,
    direction_lambda: float = 1e-3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """GEARS-style Autofocus direction-aware loss in expression space.

    The autofocus term raises absolute expression error to ``2 + gamma`` so
    larger gene-level errors receive more weight. The optional direction-aware
    term follows the GEARS paper and penalizes sign disagreement relative to
    the control mean.
    """

    error = (prediction_x0 - target_x0).abs().pow(2.0 + gamma)
    if weights is None:
        autofocus = error.mean()
    else:
        if weights.shape != error.shape:
            raise ValueError("weights must match prediction/target shape")
        per_sample_weight = weights.sum(dim=-1).clamp_min(eps)
        autofocus = (error * weights).sum(dim=-1).div(per_sample_weight).mean()

    if control_mean is None or direction_lambda <= 0:
        return autofocus
    if control_mean.ndim != 1 or control_mean.shape[0] != target_x0.shape[-1]:
        raise ValueError("control_mean must be a 1D tensor with one value per gene")

    centered_prediction = prediction_x0 - control_mean.unsqueeze(0)
    centered_target = target_x0 - control_mean.unsqueeze(0)
    direction_disagreement = (
        torch.sign(centered_target) - torch.sign(centered_prediction)
    ).square()
    if weights is None:
        direction = direction_disagreement.mean()
    else:
        per_sample_weight = weights.sum(dim=-1).clamp_min(eps)
        direction = (direction_disagreement * weights).sum(dim=-1).div(per_sample_weight).mean()
    return autofocus + direction_lambda * direction
