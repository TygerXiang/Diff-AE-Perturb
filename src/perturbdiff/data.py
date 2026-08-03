from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CONTROL_PATTERNS = ("control", "non-targeting", "non_targeting", "neg_ctrl", "ntc")


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def read_metadata_table(path: str | Path) -> list[dict[str, str]]:
    """Read CSV/TSV metadata, including gzip-compressed files."""

    path = Path(path)
    delimiter = "\t" if path.name.endswith((".tsv", ".tsv.gz")) else ","
    with _open_text(path) as handle:
        text = handle.read()
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    lines = text.splitlines()
    if not lines:
        return []
    # Some auxiliary .txt tables are actually tab-delimited despite the suffix.
    if delimiter == "," and len(lines[0].split("\t")) > len(lines[0].split(",")):
        delimiter = "\t"
    reader = csv.DictReader(lines, delimiter=delimiter)
    rows: list[dict[str, str]] = []
    for row in reader:
        cleaned: dict[str, str] = {}
        for key, value in row.items():
            clean_key = "" if key is None else str(key).strip().lstrip("\ufeff")
            cleaned[clean_key] = "" if value is None else str(value).strip()
        rows.append(cleaned)
    return rows


def read_vector(path: str | Path) -> list[str]:
    with _open_text(path) as handle:
        return [line.rstrip("\n").split("\t")[0] for line in handle]


def is_control_label(
    label: str,
    *,
    control_labels: Iterable[str] = CONTROL_PATTERNS,
) -> bool:
    lowered = str(label).strip().lower()
    return any(pattern.lower() in lowered for pattern in control_labels)


def extract_target_label(label: str, *, strategy: str = "first_token") -> str:
    """Convert a perturbation/guide label into a target name."""

    value = str(label).strip()
    if strategy == "raw":
        return value
    if strategy == "first_token":
        return value.split("_", 1)[0]
    raise ValueError(f"Unsupported target extraction strategy: {strategy}")


def canonical_guide_keys(label: str) -> tuple[str, ...]:
    """Return a small set of keys that align metadata, activity tables, and h5ad labels."""

    value = str(label).strip()
    if not value:
        return ("",)
    keys = [value]
    parts = value.split("_")
    if len(parts) >= 3:
        keys.append("_".join(parts[1:]))
    if len(parts) >= 2 and parts[0] == parts[1]:
        keys.append("_".join(parts[1:]))
    unique: list[str] = []
    for key in keys:
        if key and key not in unique:
            unique.append(key)
    return tuple(unique)


def load_guide_activity_lookup(
    path: str | Path,
    *,
    activity_col: str,
) -> dict[str, float]:
    """Load guide-level normalized activity values from a tabular file."""

    rows = read_metadata_table(path)
    if not rows:
        raise ValueError(f"No rows found in activity table: {path}")
    if activity_col not in rows[0]:
        raise KeyError(f"Column {activity_col!r} not found in activity table")
    label_col = next(iter(rows[0].keys()))
    lookup: dict[str, float] = {}
    for row in rows:
        label = str(row.get(label_col, "")).strip()
        if not label:
            continue
        raw_value = row.get(activity_col, "")
        if raw_value is None:
            continue
        value_text = str(raw_value).strip()
        if not value_text or value_text.upper() in {"NA", "N/A", "ND", "NAN"}:
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        gene = str(row.get("gene", "")).strip()
        for key in canonical_guide_keys(label):
            lookup[key] = value
        if gene:
            dup_label = f"{gene}_{label}"
            for key in canonical_guide_keys(dup_label):
                lookup[key] = value
    return lookup


def load_fully_matched_guide_lookup(
    paths: Sequence[str | Path],
) -> dict[str, str]:
    """Load per-guide fully matched sgRNA labels from mismatch-library tables."""

    lookup: dict[str, str] = {}
    for path_like in paths:
        rows = read_metadata_table(path_like)
        for row in rows:
            label = str(next(iter(row.values()), "")).strip()
            if not label:
                continue
            perfect = str(row.get("perfect match sgRNA", "")).strip()
            gene = str(row.get("gene", "")).strip()
            if not perfect:
                perfect = label
            candidates = list(canonical_guide_keys(label))
            if gene:
                candidates.extend(canonical_guide_keys(f"{gene}_{label}"))
            for key in candidates:
                lookup[key] = perfect
                if gene:
                    for full_key in canonical_guide_keys(f"{gene}_{perfect}"):
                        lookup.setdefault(full_key, perfect)
    return lookup


def load_guide_sequence_lookup(
    paths: Sequence[str | Path],
) -> dict[str, dict[str, str]]:
    """Load raw sequence and perfect-match reference metadata for each guide."""

    lookup: dict[str, dict[str, str]] = {}
    for path_like in paths:
        rows = read_metadata_table(path_like)
        for row in rows:
            label = str(next(iter(row.values()), "")).strip()
            if not label:
                continue
            gene = str(row.get("gene", "")).strip()
            sequence = str(row.get("sequence", "")).strip().upper().replace("U", "T")
            perfect = str(row.get("perfect match sgRNA", "")).strip() or label
            record = {
                "label": label,
                "gene": gene,
                "sequence": sequence,
                "perfect_match_sgrna": perfect,
            }
            candidates = list(canonical_guide_keys(label))
            if gene:
                candidates.extend(canonical_guide_keys(f"{gene}_{label}"))
            for key in candidates:
                lookup[key] = record
    return lookup


def build_virtual_guide_scales_from_embedding(
    perturbation_mapping: dict[str, int],
    embedding_weight: torch.Tensor,
    *,
    guide_library_tables: Sequence[str | Path],
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Build per-guide virtual s values from learned embedding norms.

    For each sgRNA guide g, define s_g = ||delta_g|| / ||delta_fm(g)|| where
    fm(g) is the fully matched sgRNA for that guide within the library tables.
    Guides that are themselves fully matched get s = 1.0.
    """

    if not guide_library_tables:
        raise ValueError("guide_library_tables are required to build virtual guide scales from embedding")
    if embedding_weight.ndim != 2:
        raise ValueError("embedding_weight must be a 2D tensor")
    if embedding_weight.shape[0] != len(perturbation_mapping) + 1:
        raise ValueError("embedding_weight row count does not match perturbation_mapping")

    fully_matched_lookup = load_fully_matched_guide_lookup(guide_library_tables)
    guide_sequence_lookup = load_guide_sequence_lookup(guide_library_tables)
    key_to_id: dict[str, int] = {}
    for label, idx in perturbation_mapping.items():
        for key in canonical_guide_keys(label):
            key_to_id[key] = int(idx)

    def resolve_full_id(matched_label: str) -> int | None:
        candidates = list(canonical_guide_keys(matched_label))
        # Many library tables store a perfect-match family prefix rather than a
        # concrete sgRNA row; the canonical fully matched member is usually _00.
        if not matched_label.endswith("_00"):
            candidates.extend(canonical_guide_keys(f"{matched_label}_00"))
        for key in candidates:
            full_id = key_to_id.get(key)
            if full_id is not None:
                return full_id
        return None

    scales = torch.zeros(len(perturbation_mapping) + 1, dtype=torch.float32)
    norms = embedding_weight.detach().cpu().norm(p=2, dim=1)
    rows: list[dict[str, object]] = [
        {
            "id": 0,
            "label": "control",
            "target": "control",
            "fully_matched_label": "control",
            "fully_matched_id": 0,
            "embedding_l2_norm": float(norms[0].item()),
            "fully_matched_l2_norm": 0.0,
            "s": 0.0,
        }
    ]

    for label, idx in sorted(perturbation_mapping.items(), key=lambda item: item[1]):
        matched_label = None
        for key in canonical_guide_keys(label):
            matched_label = fully_matched_lookup.get(key)
            if matched_label is not None:
                break
        if matched_label is None:
            matched_label = "_".join(str(label).split("_")[1:]) if str(label).count("_") >= 2 else str(label)
        guide_record = None
        for key in canonical_guide_keys(label):
            guide_record = guide_sequence_lookup.get(key)
            if guide_record is not None:
                break
        full_id = resolve_full_id(str(matched_label))
        if full_id is None:
            raise KeyError(
                f"Could not resolve fully matched sgRNA {matched_label!r} for guide {label!r} into perturbation_mapping"
            )
        guide_norm = float(norms[int(idx)].item())
        full_norm = float(norms[int(full_id)].item())
        if full_norm <= 0:
            raise ValueError(
                f"Fully matched guide for {label!r} has non-positive embedding norm ({full_norm})"
            )
        s_value = guide_norm / full_norm
        scales[int(idx)] = float(s_value)
        rows.append(
            {
                "id": int(idx),
                "label": str(label),
                "target": extract_target_label(str(label), strategy="first_token"),
                "sequence": "" if guide_record is None else str(guide_record.get("sequence", "")),
                "fully_matched_label": str(matched_label),
                "fully_matched_id": int(full_id),
                "embedding_l2_norm": guide_norm,
                "fully_matched_l2_norm": full_norm,
                "s": float(s_value),
            }
        )
    return scales, rows


def encode_perturbation_labels(
    labels: Sequence[str],
    *,
    control_labels: Iterable[str] = CONTROL_PATTERNS,
    target_strategy: str = "first_token",
    existing_mapping: dict[str, int] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Map perturbation labels to integer ids, reserving id 0 for controls."""

    if existing_mapping is None:
        mapping: dict[str, int] = {}
    else:
        mapping = dict(existing_mapping)
        if any(idx == 0 for target, idx in mapping.items() if target != "__control__"):
            raise ValueError("Only the control class may use perturbation id 0")

    ids: list[int] = []
    for raw_label in labels:
        # The whole project treats control as the reference condition, so we
        # give every control-like label the same reserved id: 0.
        if is_control_label(raw_label, control_labels=control_labels):
            ids.append(0)
            continue
        # target_strategy decides whether different sgRNAs for the same target
        # should collapse into one class or stay as separate conditions.
        target = extract_target_label(raw_label, strategy=target_strategy)
        if target not in mapping:
            mapping[target] = len(mapping) + 1
        ids.append(mapping[target])
    return torch.tensor(ids, dtype=torch.long), mapping


class PerturbationDataset(Dataset):
    """Dataset yielding tensors expected by PerturbDiffusionModel."""

    def __init__(
        self,
        expression: Any,
        perturbation_ids: Sequence[int] | torch.Tensor,
        *,
        perturbation_labels: Sequence[str] | None = None,
        activity_labels: Sequence[str] | None = None,
        target_ids: Sequence[int] | torch.Tensor | None = None,
        response_proxy: Sequence[float] | torch.Tensor | None = None,
        guide_features: Sequence[Sequence[float]] | torch.Tensor | None = None,
        guide_activity_targets: Sequence[float] | torch.Tensor | None = None,
        gene_names: Sequence[str] | None = None,
    ) -> None:
        n_cells = int(expression.shape[0])
        if n_cells != len(perturbation_ids):
            raise ValueError("expression and perturbation_ids must have the same length")
        self.expression = expression
        self.perturbation_ids = torch.as_tensor(perturbation_ids, dtype=torch.long)
        self.perturbation_labels = None if perturbation_labels is None else tuple(str(v) for v in perturbation_labels)
        self.activity_labels = (
            self.perturbation_labels
            if activity_labels is None
            else tuple(str(v) for v in activity_labels)
        )
        # target_ids can be coarser than perturbation_ids. For example, several
        # sgRNAs may map to different perturbation ids but share one target id.
        self.target_ids = (
            self.perturbation_ids.clone()
            if target_ids is None
            else torch.as_tensor(target_ids, dtype=torch.long)
        )
        self.response_proxy = (
            None
            if response_proxy is None
            else torch.as_tensor(response_proxy, dtype=torch.float32)
        )
        self.guide_features = (
            None
            if guide_features is None
            else torch.as_tensor(guide_features, dtype=torch.float32)
        )
        self.guide_activity_targets = (
            None
            if guide_activity_targets is None
            else torch.as_tensor(guide_activity_targets, dtype=torch.float32)
        )
        self._gene_names = None if gene_names is None else tuple(str(name) for name in gene_names)
        if self._gene_names is not None and len(self._gene_names) != int(self.expression.shape[1]):
            raise ValueError("gene_names must match expression width")
        if self.perturbation_labels is not None and len(self.perturbation_labels) != len(self):
            raise ValueError("perturbation_labels must match expression length")
        if self.activity_labels is not None and len(self.activity_labels) != len(self):
            raise ValueError("activity_labels must match expression length")
        if len(self.target_ids) != len(self):
            raise ValueError("target_ids must match expression length")
        if self.response_proxy is not None and len(self.response_proxy) != len(self):
            raise ValueError("response_proxy must match expression length")
        if self.guide_features is not None and len(self.guide_features) != len(self):
            raise ValueError("guide_features must match expression length")
        if self.guide_activity_targets is not None and len(self.guide_activity_targets) != len(self):
            raise ValueError("guide_activity_targets must match expression length")

    def __len__(self) -> int:
        return int(self.perturbation_ids.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.expression.shape[1])

    @property
    def n_perturbations(self) -> int:
        return int(self.perturbation_ids.max().item())

    @property
    def gene_names(self) -> tuple[str, ...] | None:
        return self._gene_names

    def _row_to_tensor(self, idx: int) -> torch.Tensor:
        row = self.expression[idx]
        # anndata / scipy sparse rows expose .toarray(); numpy rows do not.
        # This small conversion lets the rest of the model always see a dense
        # 1D float tensor, which is much easier for beginners to reason about.
        if hasattr(row, "toarray"):
            row = row.toarray().squeeze(0)
        return torch.as_tensor(np.asarray(row), dtype=torch.float32).flatten()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "x0": self._row_to_tensor(idx),
            "perturbation_ids": self.perturbation_ids[idx],
            "target_ids": self.target_ids[idx],
        }
        if self.perturbation_labels is not None:
            item["perturbation_label"] = self.perturbation_labels[idx]
        if self.activity_labels is not None:
            item["activity_label"] = self.activity_labels[idx]
        if self.response_proxy is not None:
            item["response_proxy"] = self.response_proxy[idx]
        if self.guide_features is not None:
            item["guide_features"] = self.guide_features[idx]
        if self.guide_activity_targets is not None:
            item["guide_activity_targets"] = self.guide_activity_targets[idx]
        return item


def load_h5ad_perturbation_dataset(
    data_path: str | Path,
    *,
    perturbation_col: str,
    activity_label_col: str | None = None,
    response_proxy_col: str | None = None,
    control_labels: Iterable[str] = CONTROL_PATTERNS,
    target_strategy: str = "first_token",
    existing_mapping: dict[str, int] | None = None,
) -> tuple[PerturbationDataset, dict[str, int]]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Reading .h5ad files requires the 'anndata' package") from exc

    adata = ad.read_h5ad(data_path)
    if perturbation_col not in adata.obs:
        raise KeyError(f"Column '{perturbation_col}' not found in adata.obs")
    labels = [str(value) for value in adata.obs[perturbation_col].to_list()]
    if activity_label_col is None:
        activity_labels = labels
    else:
        if activity_label_col not in adata.obs:
            raise KeyError(f"Column '{activity_label_col}' not found in adata.obs")
        activity_labels = [str(value) for value in adata.obs[activity_label_col].to_list()]
    perturbation_ids, mapping = encode_perturbation_labels(
        labels,
        control_labels=control_labels,
        target_strategy=target_strategy,
        existing_mapping=existing_mapping,
    )
    target_ids, _ = encode_perturbation_labels(
        labels,
        control_labels=control_labels,
        target_strategy="first_token",
        existing_mapping=None,
    )
    response_proxy = None
    if response_proxy_col is not None:
        if response_proxy_col not in adata.obs:
            raise KeyError(f"Column '{response_proxy_col}' not found in adata.obs")
        response_proxy = adata.obs[response_proxy_col].astype(float).to_numpy()
    return (
        PerturbationDataset(
            adata.X,
            perturbation_ids,
            perturbation_labels=labels,
            activity_labels=activity_labels,
            target_ids=target_ids,
            response_proxy=response_proxy,
            gene_names=adata.var_names.astype(str).to_list(),
        ),
        mapping,
    )


def load_10x_perturbation_dataset(
    data_dir: str | Path,
    *,
    metadata_path: str | Path,
    perturbation_col: str,
    barcode_col: str = "cell_barcode",
    response_proxy_col: str | None = None,
    control_labels: Iterable[str] = CONTROL_PATTERNS,
    target_strategy: str = "first_token",
) -> tuple[PerturbationDataset, dict[str, int]]:
    try:
        from scipy.io import mmread
    except ImportError as exc:
        raise ImportError("Reading 10x matrix.mtx files requires the 'scipy' package") from exc

    data_dir = Path(data_dir)
    matrix_path = next(
        (data_dir / name for name in ("matrix.mtx", "matrix.mtx.gz") if (data_dir / name).exists()),
        None,
    )
    barcode_path = next(
        (data_dir / name for name in ("barcodes.tsv", "barcodes.tsv.gz") if (data_dir / name).exists()),
        None,
    )
    gene_path = next(
        (
            data_dir / name
            for name in ("genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz")
            if (data_dir / name).exists()
        ),
        None,
    )
    if matrix_path is None or barcode_path is None or gene_path is None:
        raise FileNotFoundError(
            "Expected matrix.mtx(.gz), barcodes.tsv(.gz), and genes/features.tsv(.gz) in data_dir"
        )

    barcodes = read_vector(barcode_path)
    gene_names = read_vector(gene_path)
    metadata = read_metadata_table(metadata_path)
    by_barcode = {row[barcode_col]: row for row in metadata if row.get(barcode_col)}

    keep_indices: list[int] = []
    labels: list[str] = []
    responses: list[float] = []
    for idx, barcode in enumerate(barcodes):
        row = by_barcode.get(barcode)
        if row is None:
            continue
        keep_indices.append(idx)
        labels.append(row[perturbation_col])
        if response_proxy_col is not None:
            responses.append(float(row[response_proxy_col]))

    if not keep_indices:
        raise ValueError("No barcodes from data_dir matched metadata")

    matrix = mmread(matrix_path).tocsr().transpose().tocsr()
    expression = matrix[keep_indices]
    perturbation_ids, mapping = encode_perturbation_labels(
        labels,
        control_labels=control_labels,
        target_strategy=target_strategy,
    )
    target_ids, _ = encode_perturbation_labels(
        labels,
        control_labels=control_labels,
        target_strategy="first_token",
    )
    response_proxy = responses if response_proxy_col is not None else None
    return (
        PerturbationDataset(
            expression,
            perturbation_ids,
            perturbation_labels=labels,
            target_ids=target_ids,
            response_proxy=response_proxy,
            gene_names=gene_names,
        ),
        mapping,
    )


def create_dataloader(
    dataset: PerturbationDataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def build_signature_gene_weights(
    gene_names: Sequence[str],
    perturbation_mapping: dict[str, int],
    *,
    summary_path: str | Path | None,
    signature_gene_weight: float = 1.0,
    base_weight: float = 1.0,
) -> torch.Tensor:
    weights = torch.full(
        (len(perturbation_mapping) + 1, len(gene_names)),
        float(base_weight),
        dtype=torch.float32,
    )
    if summary_path is None:
        return weights

    summary_file = Path(summary_path)
    if not summary_file.exists():
        return weights

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    target_models = summary.get("target_models", {})
    gene_to_idx = {str(name): idx for idx, name in enumerate(gene_names)}

    for target, target_id in perturbation_mapping.items():
        model = target_models.get(str(target))
        if not model:
            continue
        for gene in model.get("signature_genes", []):
            gene_idx = gene_to_idx.get(str(gene))
            if gene_idx is not None:
                weights[target_id, gene_idx] = float(base_weight + signature_gene_weight)
    return weights
