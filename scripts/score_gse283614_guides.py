"""Build a reproducible batch sgRNA scoring table for GSE283614.

The study is CRISPRi.  This script deliberately separates measured screen
activity from sequence-only heuristics and marks unavailable CRISPRa/i tools.
It resolves hg38 protospacer context through the UCSC sequence API.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd


def rc(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def fetch(chrom: str, pos: int, spacer: str) -> dict:
    # Try both orientations around the reported starting coordinate.  The
    # supplement does not provide strand, so the matching orientation is the
    # only defensible assignment.
    n = len(spacer)
    # One wider query is faster and robust to whether the supplement reports
    # the leftmost genomic coordinate or the guide-oriented start.
    for start in (max(pos - 40, 0),):
        end = pos + 40
        url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={quote(chrom)};start={start};end={end}"
        try:
            dna = json.load(urlopen(url, timeout=8))["dna"].upper()
        except Exception:
            continue
        for strand, seq in (("+", dna), ("-", rc(dna))):
            for offset in range(max(1, len(seq) - n - 2)):
                prot = seq[offset : offset + n]
                if prot == spacer:
                    pam = seq[offset + n : offset + n + 3]
                    return {"context_23mer": seq[offset : offset + n + 3], "pam": pam, "strand": strand, "context_start": start + offset}
    return {"context_23mer": "", "pam": "", "strand": "", "context_start": np.nan}


def parse_pos(value: object) -> tuple[str, int | None]:
    m = re.search(r"(chr[^:]+):\s*(\d+)", str(value))
    return (m.group(1), int(m.group(2))) if m else ("", None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/integrated_sgrna_table.csv"))
    ap.add_argument("--output", type=Path, default=Path("results/gse283614_tool_scores.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    df = df[df["dataset"].eq("GSE283614")].copy()
    df["sequence"] = df["sgrna_sequence"].fillna("").astype(str).str.upper()
    rows = []
    for _, row in df.iterrows():
        seq = re.sub(r"[^ACGT]", "", row["sequence"])
        chrom, pos = parse_pos(row.get("genomic_position_hg38", ""))
        hit = fetch(chrom, pos, seq) if len(seq) == 20 and pos is not None else {"context_23mer":"", "pam":"", "strand":"", "context_start":np.nan}
        gc = (seq.count("G") + seq.count("C")) / len(seq) if seq else np.nan
        pam_ok = bool(re.fullmatch(r"[ACGT]GG", hit["pam"]))
        rows.append({
            "guide_identity": row["guide_identity"], "target_gene": row["target_gene"], "guide_type": row["guide_type"],
            "sgrna_sequence": seq, "genomic_position_hg38": row.get("genomic_position_hg38", ""),
            "relative_activity_mean": row.get("relative_activity_mean", np.nan),
            "relative_activity_ESC": row.get("relative_activity_ESC", np.nan), "relative_activity_DE": row.get("relative_activity_DE", np.nan),
            "gc_fraction": gc, "polyT4": int("TTTT" in seq), "homopolymer_ge4": int(bool(re.search(r"(.)\1\1\1", seq))),
            "context_23mer": hit["context_23mer"], "pam": hit["pam"], "pam_ngg": pam_ok, "strand": hit["strand"],
            "ucsc_context_status": "matched" if hit["context_23mer"] else "unresolved",
            "CRISPRi_screen": "measured" if pd.notna(row.get("relative_activity_mean", np.nan)) else "missing",
            "Azimuth_RuleSet2": "not_run_python2_legacy", "CRISPRon": "not_run_no_local_model",
            "CRISPRoff": "not_run_no_local_model", "DeepCRISPR": "not_run_no_local_model", "CRISPOR": "not_run_connection_closed", "GuideScan2": "not_run_no_local_index",
        })
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    summary = args.output.with_suffix(".summary.txt")
    summary.write_text("GSE283614 batch sgRNA scoring summary\n" + "=" * 45 + "\n" +
        f"unique guides: {len(out)}\n20-nt sequences: {(out.sgrna_sequence.str.len()==20).sum()}\n" +
        f"hg38 contexts matched: {(out.ucsc_context_status=='matched').sum()}\nNGG PAMs: {out.pam_ngg.sum()}\n" +
        f"measured CRISPRi activity: {out.relative_activity_mean.notna().sum()}\n" +
        "Sequence-only columns are heuristics, not CRISPRi efficacy predictions.\n", encoding="utf-8")
    print(out.to_string(index=False))
    print(f"wrote {args.output}\n{summary}")


if __name__ == "__main__":
    main()
