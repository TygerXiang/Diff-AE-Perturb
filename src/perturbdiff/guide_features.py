from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .modules import make_mlp


NUCLEOTIDE_TO_ROW = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


@dataclass(frozen=True)
class GuideFeatureRecord:
    label: str
    gene: str
    sequence: str = ""
    genome_input: str = ""
    sgrna_input: str = ""
    source: str = ""


def canonical_guide_keys(label: str) -> tuple[str, ...]:
    """Return matching keys for guide labels used in metadata and h5ad obs."""

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
    unique: list[str] = []
    for key in keys:
        if key and key not in unique:
            unique.append(key)
    return tuple(unique)


def _normalize_sequence(value: object) -> str:
    return str(value or "").strip().upper().replace("U", "T")


def read_activity_cnn_records(path: str | Path) -> list[GuideFeatureRecord]:
    """Read notebook-style Table_S8 activity CNN inputs."""

    path = Path(path)
    records: list[GuideFeatureRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        label_col = fieldnames[0] if fieldnames else ""
        for row in reader:
            label = str(row.get(label_col, "")).strip()
            genome_input = _normalize_sequence(row.get("genome input"))
            sgrna_input = _normalize_sequence(row.get("sgRNA input"))
            if not label or not genome_input or not sgrna_input:
                continue
            records.append(
                GuideFeatureRecord(
                    label=label,
                    gene=str(row.get("gene", "")).strip(),
                    sequence=_normalize_sequence(row.get("sgRNA sequence")),
                    genome_input=genome_input,
                    sgrna_input=sgrna_input,
                    source=path.name,
                )
            )
    return records


def load_guide_feature_records(paths: Iterable[str | Path]) -> dict[str, GuideFeatureRecord]:
    by_key: dict[str, GuideFeatureRecord] = {}
    for path_like in paths:
        for record in read_activity_cnn_records(path_like):
            for key in canonical_guide_keys(record.label):
                by_key[key] = record
            if record.gene:
                for key in canonical_guide_keys(f"{record.gene}_{record.label}"):
                    by_key[key] = record
    return by_key


def find_record_for_label(
    label: str,
    records_by_key: dict[str, GuideFeatureRecord],
) -> GuideFeatureRecord | None:
    for key in canonical_guide_keys(label):
        record = records_by_key.get(key)
        if record is not None:
            return record
    return None


def binarize_sequence(sequence: str, *, sequence_length: int = 26) -> np.ndarray:
    seq = _normalize_sequence(sequence)
    if len(seq) != sequence_length:
        raise ValueError(f"Expected {sequence_length}nt sequence, got length={len(seq)} for {seq!r}")
    arr = np.zeros((4, sequence_length), dtype=np.float32)
    for idx, base in enumerate(seq):
        row = NUCLEOTIDE_TO_ROW.get(base)
        if row is None:
            raise ValueError(f"Unsupported nucleotide {base!r} in sequence {seq!r}")
        arr[row, idx] = 1.0
    return arr


def build_notebook_cnn_feature_tensor(
    record: GuideFeatureRecord,
    *,
    sequence_length: int = 26,
) -> np.ndarray:
    if not record.genome_input or not record.sgrna_input:
        raise ValueError(f"Guide record {record.label!r} is missing CNN input sequences")
    return np.stack(
        (
            binarize_sequence(record.genome_input, sequence_length=sequence_length),
            binarize_sequence(record.sgrna_input, sequence_length=sequence_length),
        ),
        axis=2,
    ).astype(np.float32, copy=False)


def guide_cnn_feature_shape(*, sequence_length: int = 26) -> tuple[int, int, int]:
    return (4, sequence_length, 2)


def guide_cnn_feature_dim(*, sequence_length: int = 26) -> int:
    shape = guide_cnn_feature_shape(sequence_length=sequence_length)
    return int(shape[0] * shape[1] * shape[2])


class GuideStrengthEncoder(nn.Module):
    """Predict a non-negative sgRNA activity/strength from guide features."""

    def __init__(
        self,
        feature_dim: int,
        *,
        architecture: str = "cnn",
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        sequence_length: int = 26,
        cnn_channels: tuple[int, ...] = (32, 32),
        cnn_kernel_size: int = 4,
        cnn_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.architecture = architecture
        self.sequence_length = sequence_length
        if architecture == "mlp":
            self.network = make_mlp(
                feature_dim,
                hidden_dims,
                1,
                batch_norm=False,
                dropout=dropout,
                final_activation=nn.Softplus(),
            )
            return
        if architecture != "cnn":
            raise ValueError(f"Unsupported guide encoder architecture: {architecture!r}")
        expected_dim = guide_cnn_feature_dim(sequence_length=sequence_length)
        if feature_dim != expected_dim:
            raise ValueError(f"CNN guide encoder expected feature_dim={expected_dim}, got {feature_dim}")
        channels = tuple(cnn_channels) if cnn_channels else (32, 32)
        layers: list[nn.Module] = []
        in_channels = 2
        for out_channels in channels:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(4, cnn_kernel_size),
                    padding="same",
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True))
        if cnn_dropout > 0:
            layers.append(nn.Dropout(cnn_dropout))
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flat_dim = int(self.features(torch.zeros(1, 2, 4, sequence_length)).numel())
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.Sigmoid(),
            nn.Dropout(p=dropout if dropout > 0 else 0.1),
            nn.Linear(128, 32),
            nn.Sigmoid(),
            nn.Dropout(p=dropout if dropout > 0 else 0.1),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.architecture == "mlp":
            return self.network(features).squeeze(-1)
        if features.ndim == 2:
            x = features.reshape(features.shape[0], 4, self.sequence_length, 2)
        elif features.ndim == 4:
            x = features
        else:
            raise ValueError("CNN guide encoder expects flattened or 4D guide features")
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.regressor(self.features(x)).squeeze(-1)


GuideFeatureEncoder = GuideStrengthEncoder
