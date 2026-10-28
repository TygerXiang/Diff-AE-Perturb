"""Stable adapters for external CRISPRi guide activity predictors.

The training model only needs a non-negative guide activity value, while an
external predictor may return a scalar, a mapping, or the project's
``ActivityPrediction`` record.  This module keeps that boundary explicit and
lets legacy Table S8 feature models and newer sequence/context models share one
small interface.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
import math
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from .guide_activity import ActivityPrediction, GuideInput


@runtime_checkable
class GuideActivityPredictor(Protocol):
    """Protocol implemented by every guide-level activity backend.

    Backends receive the normalized ``GuideInput`` record and return
    ``ActivityPrediction`` so provenance and missing-context information are
    retained alongside the score.  A backend may be a frozen neural model, a
    ruleset, or an adapter around an external executable.
    """

    source: str

    def predict(self, guide: GuideInput) -> ActivityPrediction:
        """Predict activity for one normalized guide."""


def make_guide_input(
    guide_sequence: str | GuideInput,
    *,
    pam: str | None = None,
    target_sequence: str | None = None,
    strand: str | None = None,
    tss_distance: int | None = None,
    genomic_context: str | None = None,
    chromatin_features: tuple[float, ...] | None = None,
) -> GuideInput:
    """Build a ``GuideInput`` without duplicating context parsing in callers.

    Passing an existing ``GuideInput`` is supported for callers that already
    normalized their records.  Explicit keyword values replace only the
    corresponding non-``None`` fields, which makes it possible to add a
    sequence or TSS value to a partially populated record.
    """

    if isinstance(guide_sequence, GuideInput):
        updates: dict[str, Any] = {}
        for name, value in (
            ("pam", pam),
            ("target_sequence", target_sequence),
            ("strand", strand),
            ("tss_distance", tss_distance),
            ("genomic_context", genomic_context),
            ("chromatin_features", chromatin_features),
        ):
            if value is not None:
                updates[name] = value
        return replace(guide_sequence, **updates) if updates else guide_sequence
    return GuideInput(
        sequence=str(guide_sequence),
        pam=pam,
        target_sequence=target_sequence,
        strand=strand,
        tss_distance=tss_distance,
        genomic_context=genomic_context,
        chromatin_features=chromatin_features,
    )


def _context_mode(guide: GuideInput) -> str:
    if guide.chromatin_features is not None:
        return "chromatin_context"
    if guide.tss_distance is not None or guide.genomic_context:
        return "genome_context"
    return "sequence_only"


def _as_scalar(value: object, *, name: str) -> object:
    """Unwrap one-element numpy/torch outputs without accepting batches."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Activity prediction field {name!r} must contain one scalar")
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Activity prediction field {name!r} must contain one scalar")
        return value.reshape(-1)[0].item()
    return value


def _finite_float(value: object, *, name: str) -> float:
    value = _as_scalar(value, name=name)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Activity prediction field {name!r} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"Activity prediction field {name!r} must be finite")
    return number


def _coerce_prediction(
    value: ActivityPrediction | Mapping[str, object] | float | int | np.number,
    *,
    guide: GuideInput,
    source: str,
    normalizer: Callable[[float], float] | None,
) -> ActivityPrediction:
    """Convert common external return types into ``ActivityPrediction``."""

    if isinstance(value, ActivityPrediction):
        result = value
        if result.source:
            return result
        return replace(result, source=source)

    if isinstance(value, Mapping):
        if "score" not in value and "normalized_score" not in value:
            raise ValueError("External activity mapping must contain score or normalized_score")
        raw_value = value.get("score", value.get("normalized_score", 0.0))
        raw = _finite_float(raw_value, name="score")
        if "normalized_score" in value:
            normalized = _finite_float(value["normalized_score"], name="normalized_score")
        else:
            normalized = normalizer(raw) if normalizer is not None else raw
            normalized = _finite_float(normalized, name="normalized_score")
        result_source = str(value.get("source") or source)
        mode = str(value.get("context_mode") or _context_mode(guide))
        missing = tuple(str(item) for item in value.get("missing_features", ()))
        uncertainty = value.get("uncertainty")
        uncertainty_value = (
            None if uncertainty is None else _finite_float(uncertainty, name="uncertainty")
        )
        return ActivityPrediction(raw, normalized, result_source, mode, missing, uncertainty_value)

    raw = _finite_float(value, name="score")
    normalized = normalizer(raw) if normalizer is not None else raw
    normalized = _finite_float(normalized, name="normalized_score")
    return ActivityPrediction(raw, normalized, source, _context_mode(guide))


class CallableGuideActivityPredictor:
    """Adapt a callable or external object's ``predict`` method.

    Numeric outputs are interpreted as raw scores.  Pass ``normalizer`` when a
    model's native scale is not already the normalized activity scale.  The
    wrapped callable is never trained by this adapter.
    """

    def __init__(
        self,
        predictor: Callable[[GuideInput], object] | object,
        *,
        source: str = "external_predictor",
        normalizer: Callable[[float], float] | None = None,
    ) -> None:
        self._predictor = predictor
        self.source = source
        self._normalizer = normalizer

    def predict(self, guide: GuideInput) -> ActivityPrediction:
        if not isinstance(guide, GuideInput):
            raise TypeError("guide must be a GuideInput instance")
        method = getattr(self._predictor, "predict", None)
        value = method(guide) if callable(method) else self._predictor(guide)  # type: ignore[operator]
        return _coerce_prediction(
            value,
            guide=guide,
            source=self.source,
            normalizer=self._normalizer,
        )

    def predict_many(self, guides: Iterable[GuideInput]) -> list[ActivityPrediction]:
        return [self.predict(guide) for guide in guides]


class TorchFeaturePredictorAdapter:
    """Wrap a frozen torch module that consumes legacy feature tensors.

    ``feature_builder`` is intentionally injected.  The historical Table S8
    path can therefore keep its ``(4, 26, 2)`` builder while a new backend can
    use a different sequence/context representation without changing this
    adapter or the diffusion model.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        feature_builder: Callable[[GuideInput], object],
        *,
        source: str = "external_torch_predictor",
        normalizer: Callable[[float], float] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.module = module
        self.feature_builder = feature_builder
        self.source = source
        self._normalizer = normalizer
        self.device = torch.device(device) if device is not None else None
        self.module.eval()
        if self.device is not None:
            self.module.to(self.device)

    @torch.no_grad()
    def predict(self, guide: GuideInput) -> ActivityPrediction:
        if not isinstance(guide, GuideInput):
            raise TypeError("guide must be a GuideInput instance")
        features = self.feature_builder(guide)
        tensor = torch.as_tensor(features, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        elif tensor.ndim >= 1:
            tensor = tensor.unsqueeze(0)
        if self.device is not None:
            tensor = tensor.to(self.device)
        output = self.module(tensor)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Torch activity predictor must return a torch.Tensor")
        if output.numel() != 1:
            raise ValueError("Torch activity predictor must return one scalar per guide")
        raw = float(output.reshape(-1)[0].detach().cpu().item())
        return _coerce_prediction(
            raw,
            guide=guide,
            source=self.source,
            normalizer=self._normalizer,
        )

    def predict_many(self, guides: Iterable[GuideInput]) -> list[ActivityPrediction]:
        return [self.predict(guide) for guide in guides]


@torch.no_grad()
def predict_activity_tensor(
    predictor: GuideActivityPredictor,
    guides: Iterable[GuideInput],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, list[ActivityPrediction]]:
    """Return normalized activity values and preserve per-guide metadata."""

    predictions = [predictor.predict(guide) for guide in guides]
    values = torch.tensor(
        [prediction.normalized_score for prediction in predictions],
        dtype=dtype,
        device=device,
    )
    return values, predictions
