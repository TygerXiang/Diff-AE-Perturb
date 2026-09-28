#!/usr/bin/env python
"""Locate the two residual families and inspect GSE283614 supplementary xlsx files."""
import numpy as np, pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "datasets" / "GSE132080" / "41587_2019_387_MOESM3_ESM"
def read_tsv(p):
    df = pd.read_csv(p, sep="\t", dtype=str)
    df = df.rename(columns={df.columns[0]: "guide_label"}).dropna(axis=1, how="all")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return df

print("### which GSE132080 tables contain a 'genome input' column?")
for p in sorted(D.glob("*.txt")):
    try:
        head = pd.read_csv(p, sep="\t", nrows=0, dtype=str)
    except Exception as e:
        print(f"  {p.name}: unreadable ({e})"); continue
    cols = [c for c in head.columns if "genome" in c.lower() or "input" in c.lower()]
    print(f"  {p.name:52s} -> {cols}")

s2 = read_tsv(D / "Table_S2_large-scale_library_sequences_revised.txt")
s8 = read_tsv(D / "Table_S8_machine_learning_input.txt")

print("\n### search for RAN / RPS15 families")
for needle in ["131356438", "1438413"]:
    for name, df in [("S2", s2), ("S8", s8)]:
        hit = df[df["guide_label"].astype(str).str.contains(needle, na=False)]
        print(f"\n{name}: '{needle}' -> {len(hit)} rows")
        if len(hit):
            print(hit[["guide_label", "perfect match sgRNA", "sequence"]].head(6).to_string(index=False))

print("\n### any family whose label starts with RAN_+_ or RPS15_-_ in S8?")
for pref in ["RAN_+_", "RPS15_-_"]:
    hit = s8[s8["guide_label"].astype(str).str.startswith(pref)]
    print(f"  S8 {pref}: {len(hit)} rows; distinct families "
          f"{hit['perfect match sgRNA'].nunique() if len(hit) else 0}")
    if len(hit):
        print("   ", sorted(hit["perfect match sgRNA"].unique())[:6])

print("\n### GSE283614 supplementary xlsx files")
G = Path(__file__).resolve().parents[1] / "datasets" / "GSE283614"
for p in sorted(G.glob("*.xlsx")):
    try:
        xl = pd.ExcelFile(p)
        print(f"\n{p.name}  sheets={xl.sheet_names}")
        for sh in xl.sheet_names[:3]:
            df = pd.read_excel(p, sheet_name=sh, nrows=3)
            print(f"   [{sh}] cols: {list(df.columns)[:14]}")
    except Exception as e:
        print(f"\n{p.name}: {e}")
