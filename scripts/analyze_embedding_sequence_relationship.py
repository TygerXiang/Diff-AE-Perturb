from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, ttest_ind

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from perturbdiff.data import canonical_guide_keys, read_metadata_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether learned sgRNA embeddings recapitulate sequence/activity rules."
    )
    parser.add_argument("--embedding-activity-rows", type=Path, required=True)
    parser.add_argument("--ml-input-table", type=Path, required=True)
    parser.add_argument("--library-table", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if np.isfinite(out) else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_record(records: dict[str, dict[str, object]], label: str, gene: str, values: dict[str, object]) -> None:
    if not label:
        return
    existing = records.setdefault(label, {"label": label})
    for key, value in values.items():
        if value not in (None, ""):
            existing[key] = value
    keys = list(canonical_guide_keys(label))
    if gene:
        keys.extend(canonical_guide_keys(f"{gene}_{label}"))
    for key in keys:
        records[key] = existing


def load_sequence_records(ml_input_table: Path, library_tables: list[Path]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}

    for library_table in library_tables:
        for row in read_metadata_table(library_table):
            label = str(next(iter(row.values()), "")).strip()
            gene = str(row.get("gene", "")).strip()
            pos = parse_float(row.get("mismatch 1 position"))
            pairing = str(row.get("new pairing mismatch 1", "")).strip()
            perfect = str(row.get("perfect match sgRNA", "")).strip() or label
            add_record(
                records,
                label,
                gene,
                {
                    "gene": gene,
                    "sequence": str(row.get("sequence", "")).strip().upper().replace("U", "T"),
                    "perfect_match_sgrna": perfect,
                    "mismatch_position": pos,
                    "new_pairing": pairing,
                    "source": library_table.name,
                },
            )

    for row in read_metadata_table(ml_input_table):
        label = str(next(iter(row.values()), "")).strip()
        gene = str(row.get("gene", "")).strip()
        pos = parse_float(row.get("mismatch position"))
        pairing = str(row.get("new pairing", "")).strip()
        add_record(
            records,
            label,
            gene,
            {
                "gene": gene,
                "sequence": str(row.get("sgRNA sequence", "")).strip().upper().replace("U", "T"),
                "perfect_match_sgrna": str(row.get("perfect match sgRNA", "")).strip() or label,
                "mismatch_position": pos,
                "new_pairing": pairing,
                "genome_input": str(row.get("genome input", "")).strip().upper().replace("U", "T"),
                "sgrna_input": str(row.get("sgRNA input", "")).strip().upper().replace("U", "T"),
                "paper_mean_relative_gamma": parse_float(row.get("mean relative gamma")),
                "source": ml_input_table.name,
            },
        )

    return records


def resolve_record(label: str, records: dict[str, dict[str, object]]) -> dict[str, object] | None:
    for key in canonical_guide_keys(label):
        record = records.get(key)
        if record is not None:
            return record
    return None


def gc_content(sequence: str) -> float | None:
    seq = "".join(base for base in sequence.upper() if base in "ACGT")
    if not seq:
        return None
    return (seq.count("G") + seq.count("C")) / len(seq)


def mismatch_index_from_position(position: float | None) -> int | None:
    if position is None:
        return None
    # The compact library uses positions -1..-19 relative to the PAM-proximal end.
    # For a 20 nt guide, position -19 maps to index 1 and -1 maps to index 19.
    idx = 20 + int(position)
    return idx if 0 <= idx < 20 else None


def surrounding_g_count(sequence: str, position: float | None) -> int | None:
    idx = mismatch_index_from_position(position)
    if idx is None or len(sequence) < 20:
        return None
    count = 0
    for neighbor in (idx - 1, idx + 1):
        if 0 <= neighbor < 20 and sequence[neighbor].upper() == "G":
            count += 1
    return count


def first_g_matches_inputs(sequence: str, genome_input: str | None, sgrna_input: str | None) -> tuple[bool | None, int | None]:
    seq = str(sequence or "").upper()
    genome = str(genome_input or "").upper()
    sgrna = str(sgrna_input or "").upper()
    if not seq or not genome or not sgrna:
        return None, None
    start = sgrna.find(seq)
    if start < 0 or start >= len(genome):
        return None, None
    return sgrna[start] == "G" and genome[start] == "G", start


def first_g_matches_family(sequence: str, family_sequence: str | None) -> bool | None:
    if len(sequence) < 1 or not family_sequence:
        return None
    fam = str(family_sequence).upper()
    if len(fam) < 1:
        return None
    return sequence[0].upper() == "G" and fam[0] == "G"


def safe_corr(rows: Iterable[dict[str, object]], x_key: str, y_key: str, method: str) -> dict[str, object]:
    pairs = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        xf = float(x)
        yf = float(y)
        if np.isfinite(xf) and np.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return {"n": len(pairs), "stat": None, "pvalue": None}
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"n": len(pairs), "stat": None, "pvalue": None}
    stat, pvalue = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    return {"n": len(pairs), "stat": float(stat), "pvalue": float(pvalue)}


def centered_corr(rows: list[dict[str, object]], x_key: str, y_key: str, group_key: str = "family") -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row.get(x_key) is None or row.get(y_key) is None:
            continue
        groups.setdefault(str(row[group_key]), []).append(row)

    x_resid = []
    y_resid = []
    groups_used = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        x = np.asarray([float(row[x_key]) for row in group], dtype=np.float64)
        y = np.asarray([float(row[y_key]) for row in group], dtype=np.float64)
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            continue
        groups_used += 1
        x_resid.extend((x - x.mean()).tolist())
        y_resid.extend((y - y.mean()).tolist())

    if len(x_resid) < 3 or np.allclose(x_resid, x_resid[0]) or np.allclose(y_resid, y_resid[0]):
        return {"groups_used": groups_used, "n": len(x_resid), "stat": None, "pvalue": None}
    stat, pvalue = pearsonr(np.asarray(x_resid), np.asarray(y_resid))
    return {"groups_used": groups_used, "n": len(x_resid), "stat": float(stat), "pvalue": float(pvalue)}


def group_comparison(rows: list[dict[str, object]], flag_key: str, y_key: str) -> dict[str, object]:
    yes = [float(row[y_key]) for row in rows if row.get(flag_key) is True and row.get(y_key) is not None]
    no = [float(row[y_key]) for row in rows if row.get(flag_key) is False and row.get(y_key) is not None]
    out: dict[str, object] = {
        "true_n": len(yes),
        "false_n": len(no),
        "true_mean": float(np.mean(yes)) if yes else None,
        "false_mean": float(np.mean(no)) if no else None,
        "mean_difference_true_minus_false": float(np.mean(yes) - np.mean(no)) if yes and no else None,
    }
    if len(yes) >= 2 and len(no) >= 2:
        t_stat, t_p = ttest_ind(yes, no, equal_var=False)
        u_stat, u_p = mannwhitneyu(yes, no, alternative="two-sided")
        out.update(
            {
                "welch_t_stat": float(t_stat),
                "welch_t_pvalue": float(t_p),
                "mannwhitney_u": float(u_stat),
                "mannwhitney_pvalue": float(u_p),
            }
        )
    return out


def mean_by_category(rows: list[dict[str, object]], category_key: str, y_key: str) -> list[dict[str, object]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        cat = row.get(category_key)
        value = row.get(y_key)
        if cat is None or value is None:
            continue
        buckets.setdefault(str(cat), []).append(float(value))
    out = []
    for cat, values in sorted(buckets.items(), key=lambda item: item[0]):
        arr = np.asarray(values, dtype=np.float64)
        out.append({"category": cat, "n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr))})
    return out


def build_merged_rows(
    embedding_rows: list[dict[str, str]],
    sequence_records: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base_rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []
    for row in embedding_rows:
        label = str(row["label"])
        record = resolve_record(label, sequence_records)
        if record is None:
            unmatched_rows.append({"method": row.get("method", ""), "id": row.get("id", ""), "label": label})
            continue
        seq = str(record.get("sequence", "")).upper()
        genome_input = str(record.get("genome_input", "")).upper()
        sgrna_input = str(record.get("sgrna_input", "")).upper()
        first_g_matches, guide_input_start = first_g_matches_inputs(seq, genome_input, sgrna_input)
        family = str(record.get("perfect_match_sgrna", "") or record.get("label") or label)
        pos = record.get("mismatch_position")
        pos_f = float(pos) if pos is not None else None
        pam_distance = abs(pos_f) if pos_f is not None else None
        if pam_distance is None:
            position_region = "perfect_match"
        elif pam_distance <= 8:
            position_region = "pam_proximal_-1_to_-8"
        else:
            position_region = "distal_-9_to_-19"
        base_rows.append(
            {
                "method": row["method"],
                "id": int(row["id"]),
                "label": label,
                "target": row["target"],
                "family": family,
                "sequence": seq,
                "mismatch_position": pos_f,
                "pam_distance": pam_distance,
                "position_region": position_region,
                "new_pairing": str(record.get("new_pairing", "")),
                "is_rg_dt": str(record.get("new_pairing", "")) == "rG:dT",
                "gc_content": gc_content(seq),
                "genome_input": genome_input,
                "sgrna_input": sgrna_input,
                "guide_input_start": guide_input_start,
                "first_g_matches_genome": first_g_matches,
                "surrounding_g_count": surrounding_g_count(seq, pos_f),
                "activity": parse_float(row.get("activity")),
                "predicted_activity": parse_float(row.get("predicted_activity")),
                "embedding_norm": parse_float(row.get("embedding_norm")),
            }
        )

    by_method_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in base_rows:
        method = str(row["method"])
        label = str(row["label"])
        by_method_key[(method, label)] = row
        for key in canonical_guide_keys(label):
            by_method_key.setdefault((method, key), row)

    for row in base_rows:
        perfect = str(row["family"])
        method = str(row["method"])
        candidates = list(canonical_guide_keys(perfect))
        if not perfect.endswith("_00"):
            candidates.extend(canonical_guide_keys(f"{perfect}_00"))
        if row.get("target"):
            candidates.extend(canonical_guide_keys(f"{row['target']}_{perfect}"))
            if not perfect.endswith("_00"):
                candidates.extend(canonical_guide_keys(f"{row['target']}_{perfect}_00"))
        perfect_row = None
        for candidate in candidates:
            perfect_row = by_method_key.get((method, candidate))
            if perfect_row is not None:
                break
        perfect_norm = None if perfect_row is None else perfect_row.get("embedding_norm")
        perfect_activity = None if perfect_row is None else perfect_row.get("activity")
        row["fully_matched_embedding_norm"] = perfect_norm
        row["fully_matched_activity"] = perfect_activity
        row["embedding_s"] = (
            None
            if perfect_norm is None or float(perfect_norm) == 0.0 or row.get("embedding_norm") is None
            else float(row["embedding_norm"]) / float(perfect_norm)
        )
        row["relative_activity_to_fm"] = (
            None
            if perfect_activity is None or float(perfect_activity) == 0.0 or row.get("activity") is None
            else float(row["activity"]) / float(perfect_activity)
        )
        if row.get("first_g_matches_genome") is None:
            row["first_g_matches_genome"] = first_g_matches_family(
                str(row["sequence"]), None if perfect_row is None else str(perfect_row["sequence"])
            )

    return base_rows, unmatched_rows


def summarize_method(rows: list[dict[str, object]]) -> dict[str, object]:
    mismatch_rows = [row for row in rows if row.get("mismatch_position") is not None]
    distal_rows = [row for row in mismatch_rows if float(row["pam_distance"]) >= 9]
    proximal_rows = [row for row in mismatch_rows if float(row["pam_distance"]) <= 8]
    central_rows = [row for row in mismatch_rows if 6 <= float(row["pam_distance"]) <= 15]
    outcomes = ["activity", "relative_activity_to_fm", "embedding_norm", "embedding_s", "predicted_activity"]

    summary: dict[str, object] = {
        "n_rows": len(rows),
        "n_mismatch_rows": len(mismatch_rows),
        "n_families": len({str(row["family"]) for row in rows}),
        "pam_distance_correlations": {},
        "pam_distance_family_centered_correlations": {},
        "rg_dt_vs_other": {},
        "rg_dt_vs_other_proximal": {},
        "distal_gc_content_correlations": {},
        "distal_first_g_match": {},
        "surrounding_g_count_correlations": {},
        "position_region_means": {},
        "new_pairing_means": {},
    }
    for outcome in outcomes:
        summary["pam_distance_correlations"][outcome] = {
            "pearson": safe_corr(mismatch_rows, "pam_distance", outcome, "pearson"),
            "spearman": safe_corr(mismatch_rows, "pam_distance", outcome, "spearman"),
        }
        summary["pam_distance_family_centered_correlations"][outcome] = centered_corr(
            mismatch_rows, "pam_distance", outcome
        )
        summary["rg_dt_vs_other"][outcome] = group_comparison(mismatch_rows, "is_rg_dt", outcome)
        summary["rg_dt_vs_other_proximal"][outcome] = group_comparison(proximal_rows, "is_rg_dt", outcome)
        summary["distal_gc_content_correlations"][outcome] = {
            "pearson": safe_corr(distal_rows, "gc_content", outcome, "pearson"),
            "spearman": safe_corr(distal_rows, "gc_content", outcome, "spearman"),
        }
        summary["distal_first_g_match"][outcome] = group_comparison(distal_rows, "first_g_matches_genome", outcome)
        summary["surrounding_g_count_correlations"][outcome] = {
            "all_mismatches": safe_corr(mismatch_rows, "surrounding_g_count", outcome, "spearman"),
            "central_pam_distance_6_to_15": safe_corr(central_rows, "surrounding_g_count", outcome, "spearman"),
        }
        summary["position_region_means"][outcome] = mean_by_category(rows, "position_region", outcome)
        summary["new_pairing_means"][outcome] = mean_by_category(mismatch_rows, "new_pairing", outcome)
    return summary


def write_merged_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "id",
        "label",
        "target",
        "family",
        "sequence",
        "mismatch_position",
        "pam_distance",
        "position_region",
        "new_pairing",
        "is_rg_dt",
        "gc_content",
        "genome_input",
        "sgrna_input",
        "guide_input_start",
        "first_g_matches_genome",
        "surrounding_g_count",
        "activity",
        "relative_activity_to_fm",
        "predicted_activity",
        "embedding_norm",
        "fully_matched_embedding_norm",
        "embedding_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_rows = read_csv_rows(args.embedding_activity_rows)
    sequence_records = load_sequence_records(args.ml_input_table, args.library_table)
    merged, unmatched = build_merged_rows(embedding_rows, sequence_records)
    if not merged:
        raise ValueError("No embedding rows matched sequence metadata.")

    summaries = {
        method: summarize_method([row for row in merged if row["method"] == method])
        for method in sorted({str(row["method"]) for row in merged})
    }
    overall = {
        "embedding_activity_rows": str(args.embedding_activity_rows),
        "ml_input_table": str(args.ml_input_table),
        "library_tables": [str(path) for path in args.library_table],
        "n_input_embedding_rows": len(embedding_rows),
        "n_matched_rows": len(merged),
        "n_unmatched_rows": len(unmatched),
        "unmatched_labels": unmatched,
        "methods": summaries,
    }

    write_merged_csv(args.output_dir / "embedding_sequence_rows.csv", merged)
    with (args.output_dir / "unmatched_embedding_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "id", "label"])
        writer.writeheader()
        writer.writerows(unmatched)
    (args.output_dir / "summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
