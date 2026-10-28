"""General-purpose CRISPRi guide activity predictors.

The predictors in this module deliberately operate on a normalized guide record,
instead of the historical Table_S8 26-mer representation.  This makes the
activity path usable for new libraries while keeping external model adapters
small and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import csv
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class GuideInput:
    """Sequence and optional genomic context for one sgRNA."""

    sequence: str
    pam: str | None = None
    target_sequence: str | None = None
    strand: str | None = None
    tss_distance: int | None = None
    genomic_context: str | None = None
    chromatin_features: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ActivityPrediction:
    score: float
    normalized_score: float
    source: str
    context_mode: str
    missing_features: tuple[str, ...] = field(default_factory=tuple)
    uncertainty: float | None = None


def _clean_sequence(value: str) -> str:
    return re.sub(r"[^ACGT]", "", str(value or "").upper().replace("U", "T"))


def _is_missing(value: object) -> bool:
    return str(value or "").strip().upper() in {"", "NA", "N/A", "NAN", "NONE", "NULL"}


def _guide_keys(label: str) -> tuple[str, ...]:
    """Return the label spellings used by the prepared guide tables."""

    value = str(label).strip()
    if not value:
        return ("",)
    keys = [value]
    parts = value.split("_")
    if len(parts) >= 3:
        keys.append("_".join(parts[1:]))
    if len(parts) >= 2 and parts[0] == parts[1]:
        keys.append("_".join(parts[1:]))
    if value.endswith("-all"):
        keys.append(value.removesuffix("-all"))
    if "-all_" in value:
        keys.append(value.replace("-all_", "_"))
    return tuple(dict.fromkeys(key for key in keys if key))


def load_guide_input_lookup(path: str | Path) -> dict[str, GuideInput]:
    """Load normalized guide/context records from CSV or TSV tables.

    The prepared comprehensive table and external predictor exports use
    different column names, so this loader accepts a small set of documented
    aliases.  Missing context is retained in the resulting ``GuideInput`` and
    is reported by the backend rather than silently fabricated.
    """

    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No guide rows found in {path}")

    def choose(row: dict[str, str], names: tuple[str, ...]) -> str:
        for name in names:
            value = row.get(name)
            if value is not None and not _is_missing(value):
                return str(value).strip()
        return ""

    first_column = reader.fieldnames[0] if reader.fieldnames else ""
    lookup: dict[str, GuideInput] = {}
    for row in rows:
        label = choose(
            row,
            (
                "guide_identity",
                "guide_label",
                "label",
                "guide_id",
                "sgRNA_id",
                "sgrna_id",
                first_column,
            ),
        )
        sequence = choose(
            row,
            (
                "sgrna_sequence",
                "sgRNA sequence",
                "sequence",
                "protospacer",
                "target_sequence",
            ),
        )
        if not label or not sequence:
            continue
        gene = choose(row, ("target_gene", "gene", "target"))
        pam = choose(row, ("pam_sequence", "pam", "PAM")) or None
        strand = choose(row, ("strand", "target_strand")) or None
        tss_text = choose(
            row,
            (
                "tss_distance",
                "distance_to_tss",
                "tss_distance_primary",
                "distance_to_primary_tss",
            ),
        )
        try:
            tss_distance = None if not tss_text else int(float(tss_text))
        except ValueError:
            tss_distance = None
        genomic_context = choose(
            row,
            (
                "genomic_context",
                "target_sequence_with_pam",
                "context_23mer",
                "genome_window_26bp",
            ),
        ) or None
        record = GuideInput(
            sequence=sequence,
            pam=pam,
            target_sequence=choose(row, ("target_genomic_sequence", "target_sequence")) or None,
            strand=strand,
            tss_distance=tss_distance,
            genomic_context=genomic_context,
        )
        keys = list(_guide_keys(label))
        if gene:
            keys.extend(_guide_keys(f"{gene}_{label}"))
        for key in dict.fromkeys(keys):
            lookup[key] = record
    if not lookup:
        raise ValueError(f"No usable guide records found in {path}")
    return lookup


def load_guide_score_lookup(
    path: str | Path,
    *,
    score_col: str,
    label_col: str | None = None,
) -> dict[str, float]:
    """Load one precomputed external score per guide label.

    ``label_col`` is optional for convenience with the repository's prepared
    tables.  When omitted, the loader prefers guide identity columns instead
    of assuming that the first CSV column is the label (the comprehensive
    table starts with a ``dataset`` column).
    """

    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"No guide score rows found in {path}")
    if score_col not in fields:
        raise KeyError(f"Score column {score_col!r} not found in {path}")
    if label_col is not None and label_col not in fields:
        raise KeyError(f"Guide label column {label_col!r} not found in {path}")
    if label_col is None:
        label_col = next(
            (
                name
                for name in (
                    "guide_identity",
                    "guide_label",
                    "label",
                    "guide_id",
                    "sgRNA_id",
                    "sgrna_id",
                )
                if name in fields
            ),
            fields[0] if fields else None,
        )
    if not label_col:
        raise ValueError(f"Could not determine a guide label column in {path}")
    lookup: dict[str, float] = {}
    for row in rows:
        label = str(row.get(label_col, "")).strip()
        raw = str(row.get(score_col, "")).strip()
        if not label or _is_missing(raw):
            continue
        try:
            score = float(raw)
        except ValueError:
            continue
        if not math.isfinite(score):
            continue
        gene = str(row.get("target_gene", row.get("gene", row.get("target", ""))) or "").strip()
        keys = list(_guide_keys(label))
        if gene:
            keys.extend(_guide_keys(f"{gene}_{label}"))
        for key in dict.fromkeys(keys):
            lookup[key] = score
    if not lookup:
        raise ValueError(f"No finite guide scores found in {path}")
    return lookup


class CRISPRiRuleSetPredictor:
    """Transparent, sequence-plus-TSS CRISPRi baseline.

    This is a deployable baseline rather than a claim to reproduce a licensed
    vendor score.  It captures established CRISPRi design signals: guide
    sequence quality, GC balance, poly-T avoidance, and promoter distance.
    The output is in [0, 1] and is intended for calibration on measured data.
    """

    source = "crispri_ruleset_baseline"

    def predict(self, guide: GuideInput) -> ActivityPrediction:
        seq = _clean_sequence(guide.sequence)
        missing: list[str] = []
        if len(seq) != 20:
            missing.append("20nt_protospacer")
        if not guide.pam:
            missing.append("pam")
        if guide.tss_distance is None:
            missing.append("tss_distance")
        if not seq:
            return ActivityPrediction(0.0, 0.0, self.source, "sequence_only", tuple(missing))

        gc = sum(base in "GC" for base in seq) / len(seq)
        gc_penalty = min(abs(gc - 0.50) / 0.50, 1.0)
        poly_t_penalty = 1.0 if "TTTT" in seq else 0.0
        homopolymer_penalty = 0.35 if any(base * 5 in seq for base in "ACGT") else 0.0
        # CRISPRi guides near the TSS are generally stronger; retain a smooth
        # decay so this remains usable when a precise design window is absent.
        if guide.tss_distance is None:
            tss_score = 0.5
        else:
            tss_score = math.exp(-abs(float(guide.tss_distance)) / 250.0)
        pam_score = 1.0 if guide.pam and _clean_sequence(guide.pam).endswith("GG") else 0.65
        raw = 0.72 * tss_score + 0.18 * pam_score + 0.10 * (1.0 - gc_penalty)
        raw *= 1.0 - 0.75 * poly_t_penalty - homopolymer_penalty
        score = float(max(0.0, min(1.0, raw)))
        mode = "chromatin_context" if guide.chromatin_features is not None else (
            "genome_context" if guide.tss_distance is not None or guide.genomic_context else "sequence_only"
        )
        return ActivityPrediction(score, score, self.source, mode, tuple(missing))


class FrozenActivityCalibrator(nn.Module):
    """Small monotonic calibrator for frozen external activity scores.

    ``raw_score`` is detached by default, so training PerturbDiff cannot
    accidentally fine-tune an external predictor through the calibration path.
    """

    def __init__(self, *, init_scale: float = 1.0, init_bias: float = 0.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(float(math.log(max(init_scale, 1e-4)))))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, raw_score: torch.Tensor, *, detach_input: bool = True) -> torch.Tensor:
        if raw_score.ndim != 1:
            raise ValueError("raw_score must have shape (batch,)")
        value = raw_score.detach() if detach_input else raw_score
        scale = torch.exp(self.log_scale).clamp_min(1e-4)
        # Center the positive transform at zero so a control score of exactly
        # zero remains zero after calibration.  This matters because controls
        # are the reference condition for both the explicit activity channel
        # and the target-specific doser.
        bias_value = torch.nn.functional.softplus(self.bias)
        return torch.nn.functional.softplus(value * scale + self.bias) - bias_value


def predict_activity_batch(
    guides: list[GuideInput],
    predictor: CRISPRiRuleSetPredictor | None = None,
) -> list[ActivityPrediction]:
    """Score a batch while preserving per-guide provenance metadata."""

    predictor = predictor or CRISPRiRuleSetPredictor()
    return [predictor.predict(guide) for guide in guides]
