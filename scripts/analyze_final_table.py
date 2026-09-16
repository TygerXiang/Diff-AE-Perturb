#!/usr/bin/env python
"""
Final validation + analysis of the comprehensive sgRNA table.

Writes results/final_summary_report.txt and prints the same content.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

WORKSPACE = Path(__file__).resolve().parents[1]
TABLE = WORKSPACE / "results/comprehensive_sgrna_table.csv"
REPORT = WORKSPACE / "results/final_summary_report.txt"

L = []
def p(s=""):
    print(s)
    L.append(s)


def corr(df, x, y, label):
    d = df[[x, y]].dropna()
    if len(d) < 8:
        p(f"  {label:46s} n={len(d):3d}  (too few)")
        return
    if d[x].nunique() < 2 or d[y].nunique() < 2:
        p(f"  {label:46s} n={len(d):3d}  (no variance in this subset)")
        return
    r, pr = pearsonr(d[x], d[y])
    rho, ps = spearmanr(d[x], d[y])
    p(f"  {label:46s} n={len(d):3d}  r={r:+.3f} (p={pr:.1e})  rho={rho:+.3f} (p={ps:.1e})")


def main():
    df = pd.read_csv(TABLE)
    real = df[~df["guide_identity"].isin(["control", "*"])].copy()

    p("=" * 78)
    p("COMPREHENSIVE sgRNA TABLE -- FINAL VALIDATION")
    p("=" * 78)
    p(f"file   : {TABLE}")
    p(f"rows   : {len(df)}   columns: {len(df.columns)}")
    p("")
    p("by dataset:")
    for k, v in df["dataset"].value_counts().items():
        p(f"  {k:12s} {v}")

    p("")
    p("-" * 78)
    p("1. FIELD COMPLETENESS (real guides only, n=%d)" % len(real))
    p("-" * 78)
    fields = ["sgrna_sequence", "target_genomic_sequence", "pam_sequence",
              "reference_source", "n_mismatches", "mismatch_positions_from_pam",
              "mismatch_distance_to_pam", "min_distance_to_pam",
              "embedding_norm", "virtual_scale",
              "mean_relative_gamma", "relative_activity_mean"]
    for c in fields:
        if c not in real.columns:
            continue
        n = real[c].notna().sum()
        p(f"  {c:28s} {n:4d}/{len(real)}  ({100*n/len(real):5.1f}%)  {'#'*int(40*n/len(real))}")

    p("")
    p("-" * 78)
    p("2. UNRESOLVED ROWS -- and why")
    p("-" * 78)
    miss = real[real["target_genomic_sequence"].isna()]
    if len(miss) == 0:
        p("  none")
    else:
        for r in miss.to_dict("records"):
            p(f"  {r['dataset']:10s} {str(r['guide_identity']):22s} "
              f"type={r.get('guide_type')}  -> no genomic target "
              f"(non-targeting control / no coordinate)")

    p("")
    p("-" * 78)
    p("3. GSE132080 SEQUENCE & MISMATCH VALIDATION")
    p("-" * 78)
    g = real[real["dataset"] == "GSE132080"]
    p(f"  guides                              : {len(g)}")
    p(f"  reference protospacer (Table S2 _00): {g['target_genomic_sequence'].notna().sum()}")
    p(f"  PAM 3-mer available                 : {g['pam_sequence'].notna().sum()}")
    v = g["mismatch_validated_vs_table_s2"].dropna()
    p(f"  mismatch call == published Table S2 : {int(v.sum())}/{len(v)} "
      f"({100*v.mean():.1f}%)")
    p("")
    p("  PAM composition:")
    for k, n in g["pam_sequence"].value_counts().items():
        p(f"    {k}: {n}")
    p("")
    p("  mismatch count:")
    for k, n in g["n_mismatches"].value_counts().sort_index().items():
        p(f"    {int(k)} mismatch(es): {n}")
    p("")
    d = g["min_distance_to_pam"].dropna()
    p(f"  distance from mismatch to PAM: mean={d.mean():.2f}  median={d.median():.0f}  "
      f"range={d.min():.0f}-{d.max():.0f}")
    p("  distribution:")
    bins = [-1, 4, 9, 14, 19]
    labels = [" 0-4  (PAM-proximal)", " 5-9 ", "10-14", "15-19 (PAM-distal)"]
    cut = pd.cut(d, bins=bins, labels=labels)
    for lab in labels:
        c = int((cut == lab).sum())
        p(f"    {lab:22s} {c:4d}  ({100*c/len(d):5.1f}%)  {'#'*int(30*c/len(d))}")

    p("")
    p("-" * 78)
    p("4. GSE283614 SEQUENCE COVERAGE")
    p("-" * 78)
    h = real[real["dataset"] == "GSE283614"]
    p(f"  guides                              : {len(h)}")
    p(f"  with hg38 target sequence + PAM     : {h['target_genomic_sequence'].notna().sum()}")
    p(f"  targeting guides resolved           : "
      f"{(h['guide_type'].isin(['enhancer','promoter']) & h['pam_sequence'].notna()).sum()}"
      f"/{h['guide_type'].isin(['enhancer','promoter']).sum()}")

    p("")
    p("-" * 78)
    p("5. CORRELATIONS (GSE132080, experimental gamma)")
    p("-" * 78)
    corr(g, "virtual_scale", "mean_relative_gamma", "model virtual_scale  vs activity")
    corr(g, "embedding_norm", "mean_relative_gamma", "model embedding norm vs activity")
    corr(g, "min_distance_to_pam", "mean_relative_gamma", "mismatch->PAM distance vs activity")
    corr(g, "n_mismatches", "mean_relative_gamma", "n mismatches          vs activity")
    corr(g, "sgrna_gc_content", "mean_relative_gamma", "sgRNA GC content      vs activity")

    p("")
    p("-" * 78)
    p("6. CORRELATIONS (GSE283614, relative activity)")
    p("-" * 78)
    corr(h, "sgrna_gc_content", "relative_activity_mean", "sgRNA GC content vs activity")

    p("")
    p("=" * 78)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
