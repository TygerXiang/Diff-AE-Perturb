from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


GUIDE_CAPTURE_TYPE = "CRISPR Guide Capture"
GENE_EXPRESSION_TYPE = "Gene Expression"

OCT4_DE_GUIDES = (
    "OCT4_1-38",
    "OCT4_1-78",
    "OCT4_1-37",
    "OCT4_1-74",
    "OCT4_1-17",
    "OCT4_1-55",
    "OCT4_1-73",
)
OCT4_PE_GUIDES = (
    "OCT4_1-83",
    "OCT4_1-46",
    "OCT4_1-47",
    "OCT4_1-21",
    "OCT4_1-41",
    "OCT4_1-62",
    "OCT4_1-81",
)
NANOG_E2_GUIDES = (
    "NANOG-1-4",
    "NANOG-1-5",
    "NANOG-4-22",
    "NANOG-4-1",
    "NANOG-4-6",
    "NANOG-4-10",
)
NANOG_E1_GUIDES = (
    "NANOG_1-23",
    "NANOG_1-1",
    "NANOG_1-2",
    "NANOG_1-3",
    "NANOG_1-9",
    "NANOG_1-6",
    "NANOG_1-21",
)
SAFE_HARBOR_GUIDES = (
    "SafeHarborchr1_1007",
    "SafeHarborchr10_1072",
    "SafeHarborchr11_443",
    "SafeHarborchr2_1008",
    "SafeHarborchr12_290",
    "SafeHarborchr12_26",
    "SafeHarborchr6_1081",
    "SafeHarborchr6_237",
)

TABLE_S1_REGION_GUIDES = {
    ("OCT4", "OCT4-DE"): OCT4_DE_GUIDES,
    ("OCT4", "OCT4-PE"): OCT4_PE_GUIDES,
    ("NANOG", "NANOG e1"): NANOG_E1_GUIDES,
    ("NANOG", "NANOG e2"): NANOG_E2_GUIDES,
}


@dataclass(frozen=True)
class GuideCallConfig:
    min_total_umis: int = 2
    min_top_umis: int = 2
    min_top_fraction: float = 0.5
    min_top_ratio: float = 1.5


def normalize_guide_label(label: str) -> str:
    value = str(label).strip()
    if not value:
        return ""
    value = value.replace("+", "_")
    value = value.replace("NT_Control-", "NT_Control_")
    value = re.sub(r"(?<=pro)-(?=\d)", "", value)
    value = re.sub(r"__+", "_", value)
    return value


def guide_family(label: str) -> str:
    value = normalize_guide_label(label)
    upper = value.upper()
    if upper.startswith("NT_CONTROL") or upper.startswith("NT-"):
        return "control_nt"
    if upper.startswith("SAFEHARBOR") or upper.startswith("ST-"):
        return "control_safeharbor"
    if upper.startswith("OCT4"):
        return "OCT4"
    if upper.startswith("NANOG"):
        return "NANOG"
    if "_" in value:
        return value.split("_", 1)[0]
    if "-" in value:
        return value.split("-", 1)[0]
    return value


def guide_target_gene(label: str) -> str:
    family = guide_family(label)
    if family == "OCT4":
        return "POU5F1"
    if family == "NANOG":
        return "NANOG"
    if family.startswith("control"):
        return "control"
    return family


def guide_type(label: str) -> str:
    value = normalize_guide_label(label).lower()
    if value.startswith("nt_control") or value.startswith("nt-"):
        return "non_targeting_control"
    if value.startswith("st-") or value.startswith("safeharbor"):
        return "safe_harbor_control"
    if "pro" in value:
        return "promoter"
    if value:
        return "enhancer"
    return "unknown"


def _find_header_row(raw: pd.DataFrame) -> int:
    for idx in range(raw.shape[0]):
        values = {str(value).strip() for value in raw.iloc[idx].tolist()}
        if {"Type", "Region", "Group", "Name"}.issubset(values):
            return idx
    raise ValueError("Could not find the Table S1 header row containing Type/Region/Group/Name")


def read_table_s1_grna_library(path: str | Path, *, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read GSE283614 Table S1 gRNA library metadata from the supplementary xlsx."""

    try:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    except ImportError as exc:
        raise ImportError("Reading GSE283614 Table S1 requires openpyxl") from exc
    header_idx = _find_header_row(raw)
    table = raw.iloc[header_idx + 1 :].copy()
    table.columns = [str(value).strip() for value in raw.iloc[header_idx].tolist()]
    table = table.dropna(how="all")
    table = table[table["Name"].notna()].copy()
    rename = {
        "Type": "table_s1_type",
        "Region": "region",
        "Group": "group",
        "Name": "table_s1_name",
        "Sequence": "sequence",
        "Starting position (hg38)": "starting_position_hg38",
        "Z-score, ESC": "screen_z_score_ESC",
        "Z-score, DE": "screen_z_score_DE",
    }
    table = table.rename(columns=rename)
    keep_cols = [col for col in rename.values() if col in table.columns]
    table = table[keep_cols].copy()
    for col in ("table_s1_type", "region", "group", "table_s1_name", "sequence", "starting_position_hg38"):
        if col in table.columns:
            table[col] = table[col].fillna("").astype(str).str.strip()
    for col in ("screen_z_score_ESC", "screen_z_score_DE"):
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    return table.reset_index(drop=True)


def table_s1_name_to_guide_identity(row: pd.Series) -> tuple[str, str]:
    """Map a Table S1 gRNA row to the normalized 10x guide capture label."""

    name = str(row.get("table_s1_name", "")).strip()
    table_type = str(row.get("table_s1_type", "")).strip()
    region = str(row.get("region", "")).strip()
    if not name:
        return "", "missing_name"
    if name.startswith("NT-"):
        suffix = name.split("-", 1)[1]
        return f"NT_Control_{suffix}", "control_name"
    if name.startswith("ST-"):
        try:
            idx = int(name.split("-", 1)[1]) - 1
        except ValueError:
            return "", "safe_harbor_name_error"
        if 0 <= idx < len(SAFE_HARBOR_GUIDES):
            return SAFE_HARBOR_GUIDES[idx], "safe_harbor_order"
        return "", "safe_harbor_out_of_range"
    if "promoter" in region.lower() or "pro" in name.lower():
        return normalize_guide_label(name), "promoter_name"

    region_guides = TABLE_S1_REGION_GUIDES.get((table_type, region))
    if region_guides is None:
        return normalize_guide_label(name), "normalized_name_fallback"
    region_rows = [match for match in re.finditer(rf"^{re.escape(region.split()[-1])}-(\d+)$", name)]
    if region_rows:
        order_idx = int(region_rows[0].group(1)) - 1
    else:
        order_match = re.search(r"-(\d+)$", name)
        if order_match is None and table_type in {"OCT4", "NANOG"}:
            order_match = re.search(r"(\d+)$", name)
        if order_match is None:
            return "", "region_order_error"
        order_idx = int(order_match.group(1)) - 1
    if 0 <= order_idx < len(region_guides):
        return region_guides[order_idx], "region_order"
    return "", "region_order_out_of_range"


def _normalized_header_value(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _find_header_row_with_columns(raw: pd.DataFrame, required: set[str]) -> int:
    normalized_required = {_normalized_header_value(value) for value in required}
    for idx in range(raw.shape[0]):
        values = {_normalized_header_value(value) for value in raw.iloc[idx].tolist()}
        if normalized_required.issubset(values):
            return idx
    raise ValueError(f"Could not find a header row containing: {sorted(required)}")


def read_log2fc_reference_grna_table(
    path: str | Path,
    *,
    sheet_name: int | str = "Individual gRNAs",
) -> pd.DataFrame:
    """Read reference CRISPRi screen individual-gRNA log2FC values."""

    try:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    except ImportError as exc:
        raise ImportError("Reading reference log2FC workbooks requires openpyxl") from exc
    header_idx = _find_header_row_with_columns(
        raw,
        {
            "Region",
            "gRNA name",
            "gRNA sequence",
            "Log2(Day 9/Day 0)",
        },
    )
    table = raw.iloc[header_idx + 1 :].copy()
    table.columns = [_normalized_header_value(value) for value in raw.iloc[header_idx].tolist()]
    table = table.dropna(how="all")
    rename = {
        "Region": "reference_region",
        "gRNA name": "reference_grna_name",
        "gRNA sequence": "sequence",
        "gRNA coordinates (hg38)": "reference_coordinates_hg38",
        "Raw reads Day 0": "raw_reads_day0",
        "Raw reads Day 9": "raw_reads_day9",
        "Normalization to total reads Day 0": "normalized_reads_day0",
        "Normalization to total reads Day 9": "normalized_reads_day9",
        "Fold change Day 9/Day 0": "fold_change_day9_day0",
        "Log2(Day 9/Day 0)": "screen_log2fc_day9_day0",
        "Average of abundance": "average_abundance",
        "Log2(Average of abundance)": "log2_average_abundance",
        "Z-score (individual-safeharbor average)/STD": "reference_z_score",
    }
    table = table.rename(columns=rename)
    keep_cols = [col for col in rename.values() if col in table.columns]
    table = table[keep_cols].copy()
    for col in ("reference_region", "reference_grna_name", "sequence", "reference_coordinates_hg38"):
        if col in table.columns:
            table[col] = table[col].fillna("").astype(str).str.strip()
    for col in (
        "raw_reads_day0",
        "raw_reads_day9",
        "normalized_reads_day0",
        "normalized_reads_day9",
        "fold_change_day9_day0",
        "screen_log2fc_day9_day0",
        "average_abundance",
        "log2_average_abundance",
        "reference_z_score",
    ):
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    table["sequence"] = table["sequence"].str.upper().str.replace("U", "T", regex=False)
    return table.reset_index(drop=True)


def _add_log2fc_relative_activity(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["screen_depletion_effect"] = -pd.to_numeric(out["screen_log2fc_day9_day0"], errors="coerce")
    references: dict[str, pd.Series] = {}
    for region, group in out.groupby("reference_region", dropna=False):
        positive = group[pd.to_numeric(group["screen_depletion_effect"], errors="coerce") > 0]
        if positive.empty:
            continue
        idx = positive["screen_depletion_effect"].idxmax()
        references[str(region)] = out.loc[idx]

    out["activity_reference_effect"] = np.nan
    out["activity_reference_log2fc"] = np.nan
    out["activity_reference_grna_name"] = ""
    for idx, row in out.iterrows():
        reference = references.get(str(row["reference_region"]))
        if reference is None:
            continue
        out.at[idx, "activity_reference_effect"] = float(reference["screen_depletion_effect"])
        out.at[idx, "activity_reference_log2fc"] = float(reference["screen_log2fc_day9_day0"])
        out.at[idx, "activity_reference_grna_name"] = str(reference["reference_grna_name"])
    out["relative_activity_signed"] = out["screen_depletion_effect"] / out["activity_reference_effect"]
    out["relative_activity"] = out["relative_activity_signed"].clip(lower=0)
    return out


def build_log2fc_relative_activity_table(
    path: str | Path,
    *,
    guide_library_path: str | Path | None = None,
    sheet_name: int | str = "Individual gRNAs",
) -> pd.DataFrame:
    """Build relative activity from reference-screen log2FC values.

    The source workbook provides log2(day9/day0), so it is closer to the
    literature activity definition than z-scores. With a constant number of cell
    doublings, gamma_variant / gamma_reference reduces to a log2FC ratio. For a
    depletion screen, this function uses -log2FC as the positive effect proxy
    and uses the strongest depletion guide in the same reference region as the
    default denominator.

    If guide_library_path is provided, rows are left-joined to the GSE283614
    library by sgRNA sequence and returned with guide_identity as the first
    column, ready for --activity-source direct.
    """

    reference = _add_log2fc_relative_activity(
        read_log2fc_reference_grna_table(path, sheet_name=sheet_name)
    )
    if guide_library_path is None:
        return reference

    guide_table = read_table_s1_grna_library(guide_library_path)
    mapped = guide_table.apply(table_s1_name_to_guide_identity, axis=1, result_type="expand")
    guide_table.insert(0, "guide_identity", mapped.iloc[:, 0].astype(str))
    guide_table.insert(1, "mapping_strategy", mapped.iloc[:, 1].astype(str))
    guide_table["guide_family"] = guide_table["guide_identity"].map(guide_family)
    guide_table["target_gene"] = guide_table["guide_identity"].map(guide_target_gene)
    guide_table["guide_type"] = guide_table["guide_identity"].map(guide_type)
    guide_table["sequence"] = guide_table["sequence"].fillna("").astype(str).str.upper().str.replace("U", "T", regex=False)

    reference_by_sequence = reference.drop_duplicates("sequence", keep="first")
    merged = guide_table.merge(
        reference_by_sequence,
        on="sequence",
        how="left",
        suffixes=("", "_reference"),
    )
    control_mask = merged["guide_family"].astype(str).str.startswith("control")
    for col in ("relative_activity", "relative_activity_signed", "screen_depletion_effect"):
        merged.loc[control_mask, col] = 0.0
    first_cols = [
        "guide_identity",
        "relative_activity",
        "relative_activity_signed",
        "screen_log2fc_day9_day0",
        "screen_depletion_effect",
        "activity_reference_log2fc",
        "activity_reference_effect",
        "activity_reference_grna_name",
        "reference_region",
        "reference_grna_name",
        "mapping_strategy",
        "guide_family",
        "target_gene",
        "guide_type",
        "sequence",
    ]
    return merged[[col for col in first_cols if col in merged.columns]]


def _add_relative_activity(
    table: pd.DataFrame,
    state: str,
    *,
    group_cols: Sequence[str] = ("table_s1_type", "region"),
) -> None:
    effect_col = f"screen_depletion_effect_{state}"
    reference_col = f"activity_reference_effect_{state}"
    signed_col = f"relative_activity_signed_{state}"
    relative_col = f"relative_activity_{state}"
    effects = pd.to_numeric(table[effect_col], errors="coerce")
    group_keys = [col for col in group_cols if col in table.columns]
    if group_keys:
        references = table.assign(**{effect_col: effects}).groupby(group_keys, dropna=False)[effect_col].transform("max")
    else:
        references = pd.Series(np.repeat(effects.max(skipna=True), len(table)), index=table.index)
    references = references.where(references > 0)
    table[reference_col] = references
    table[signed_col] = effects / references
    table[relative_col] = table[signed_col].clip(lower=0)


def build_table_s1_activity_table(path: str | Path, *, sheet_name: int | str = 0) -> pd.DataFrame:
    """Create a guide activity table aligned to GSE283614 guide-capture labels.

    Table S1 z-scores summarize log2FC(day 9/day 0) screen enrichment. The
    original sgRNA-activity definition is a ratio of growth phenotypes
    gamma_variant / gamma_original. Because Table S1 exposes z-scored log2FC
    rather than raw counts, this function treats -z as a standardized depletion
    phenotype proxy and divides each guide by a within-region reference effect.

    In the absence of an explicit "original sgRNA" annotation for GSE283614, the
    default reference is the strongest depletion effect within each
    (Type, Region) group. If a curated original-guide map becomes available, use
    that denominator instead of this strongest-guide proxy.
    """

    table = read_table_s1_grna_library(path, sheet_name=sheet_name)
    mapped = table.apply(table_s1_name_to_guide_identity, axis=1, result_type="expand")
    table.insert(0, "guide_identity", mapped.iloc[:, 0].astype(str))
    table.insert(1, "mapping_strategy", mapped.iloc[:, 1].astype(str))
    table["guide_family"] = table["guide_identity"].map(guide_family)
    table["target_gene"] = table["guide_identity"].map(guide_target_gene)
    table["guide_type"] = table["guide_identity"].map(guide_type)
    table["gene"] = table["target_gene"]
    for state in ("ESC", "DE"):
        z_col = f"screen_z_score_{state}"
        strength_col = f"screen_strength_{state}"
        effect_col = f"screen_depletion_effect_{state}"
        table[effect_col] = -pd.to_numeric(table[z_col], errors="coerce")
        table[strength_col] = table[effect_col].clip(lower=0)
        _add_relative_activity(table, state)
    table["screen_strength_mean"] = table[["screen_strength_ESC", "screen_strength_DE"]].mean(axis=1, skipna=True)
    table["relative_activity_mean"] = table[["relative_activity_ESC", "relative_activity_DE"]].mean(axis=1, skipna=True)
    return table


def read_tsv_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, compression="infer", dtype=str)


def _find_10x_file(data_dir: Path, names: Sequence[str], pattern: str) -> Path | None:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    matches = sorted(data_dir.glob(pattern))
    return matches[0] if matches else None


def read_state_matrices(data_dir: str | Path) -> tuple[sparse.csr_matrix, list[str], list[str], list[str], list[str]]:
    data_dir = Path(data_dir)
    matrix_path = _find_10x_file(
        data_dir,
        ("matrix.mtx", "matrix.mtx.gz"),
        "*matrix.mtx*",
    )
    barcode_path = _find_10x_file(
        data_dir,
        ("barcodes.tsv", "barcodes.tsv.gz"),
        "*barcodes.tsv*",
    )
    feature_path = _find_10x_file(
        data_dir,
        ("features.tsv", "features.tsv.gz", "features.tsv_2.gz", "genes.tsv", "genes.tsv.gz"),
        "*features.tsv*",
    )
    if matrix_path is None or barcode_path is None or feature_path is None:
        raise FileNotFoundError(
            "Expected matrix.mtx(.gz), barcodes.tsv(.gz), and features/genes.tsv(.gz) in data_dir"
        )

    matrix = mmread(matrix_path).tocsr().transpose().tocsr()
    barcodes = read_tsv_table(barcode_path).iloc[:, 0].astype(str).tolist()
    feature_table = read_tsv_table(feature_path)
    if feature_table.shape[1] < 3:
        raise ValueError(f"Feature table {feature_path} must have at least three columns")
    feature_ids = feature_table.iloc[:, 0].astype(str).tolist()
    feature_names = feature_table.iloc[:, 1].astype(str).tolist()
    feature_types = feature_table.iloc[:, 2].astype(str).tolist()
    return matrix, barcodes, feature_ids, feature_names, feature_types


def split_feature_matrix(
    matrix: sparse.csr_matrix,
    feature_types: Sequence[str],
) -> tuple[sparse.csr_matrix, list[int], sparse.csr_matrix, list[int]]:
    gene_indices = [idx for idx, value in enumerate(feature_types) if value == GENE_EXPRESSION_TYPE]
    guide_indices = [idx for idx, value in enumerate(feature_types) if value == GUIDE_CAPTURE_TYPE]
    if not gene_indices or not guide_indices:
        raise ValueError("Could not find both gene expression and guide capture features")
    return matrix[:, gene_indices].tocsr(), gene_indices, matrix[:, guide_indices].tocsr(), guide_indices


def call_guides_from_matrix(
    guide_matrix: sparse.csr_matrix,
    guide_names: Sequence[str],
    *,
    config: GuideCallConfig = GuideCallConfig(),
) -> pd.DataFrame:
    guide_matrix = guide_matrix.tocsr()
    rows: list[dict[str, object]] = []
    for cell_idx in range(guide_matrix.shape[0]):
        row = guide_matrix.getrow(cell_idx)
        data = row.data
        indices = row.indices
        total = int(data.sum()) if data.size else 0
        if total <= 0 or data.size == 0:
            rows.append(
                {
                    "guide_identity": "unassigned",
                    "perturbation_label": "unassigned",
                    "guide_family": "unassigned",
                    "target_gene": "unassigned",
                    "guide_type": "unknown",
                    "guide_assignment_status": "no_guide",
                    "guide_total_umis": 0,
                    "guide_top_umis": 0,
                    "guide_second_umis": 0,
                    "guide_top_fraction": 0.0,
                    "guide_top_ratio": 0.0,
                }
            )
            continue

        order = np.argsort(data)
        top_pos = int(order[-1])
        top_idx = int(indices[top_pos])
        top_count = int(data[top_pos])
        second_count = int(data[order[-2]]) if data.size >= 2 else 0
        top_label = normalize_guide_label(guide_names[top_idx])
        family = guide_family(top_label)
        top_fraction = float(top_count / max(total, 1))
        top_ratio = float(top_count / max(second_count, 1))
        is_control = family.startswith("control")
        assigned = (
            total >= config.min_total_umis
            and top_count >= config.min_top_umis
            and top_fraction >= config.min_top_fraction
            and top_ratio >= config.min_top_ratio
        )
        if assigned:
            status = "assigned_control" if is_control else "assigned_guide"
            perturbation_label = "control" if is_control else top_label
        else:
            status = "ambiguous"
            perturbation_label = "unassigned"
        rows.append(
            {
                "guide_identity": top_label,
                "perturbation_label": perturbation_label,
                "guide_family": "control" if is_control else family,
                "target_gene": "control" if is_control else guide_target_gene(top_label),
                "guide_type": "control" if is_control else guide_type(top_label),
                "guide_assignment_status": status,
                "guide_total_umis": total,
                "guide_top_umis": top_count,
                "guide_second_umis": second_count,
                "guide_top_fraction": top_fraction,
                "guide_top_ratio": top_ratio,
            }
        )
    return pd.DataFrame(rows)
