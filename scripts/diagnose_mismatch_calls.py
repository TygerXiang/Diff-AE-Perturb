#!/usr/bin/env python
"""Diagnose the mismatch-calling disagreement against published Table S2 columns."""
import numpy as np
import pandas as pd
from pathlib import Path

D = Path(r"C:\Users\Admin\myprojects\diff-project\datasets\GSE132080\41587_2019_387_MOESM3_ESM")

def read_tsv(p):
    df = pd.read_csv(p, sep="\t", dtype=str)
    df = df.rename(columns={df.columns[0]: "guide_label"}).dropna(axis=1, how="all")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return df

s8 = read_tsv(D / "Table_S8_machine_learning_input.txt")
s2 = read_tsv(D / "Table_S2_large-scale_library_sequences_revised.txt")

print("S2 columns:", list(s2.columns))
print("\nS2 mismatch columns populated:")
print("  mismatch 1 position:", s2["mismatch 1 position"].notna().sum(), "/", len(s2))
print("  mismatch 2 position:", s2["mismatch 2 position"].notna().sum(), "/", len(s2))

# ---- reference guide (_00) per family from S2
s2["family"] = s2["perfect match sgRNA"]
print("\n=== AARS2_+_44281027.23-P1P2 family in S2 ===")
sub = s2[s2["family"] == "AARS2_+_44281027.23-P1P2"]
cols = ["guide_label", "sequence", "mismatch 1 position", "new pairing mismatch 1",
        "mismatch 2 position", "new pairing mismatch 2"]
print(sub[cols].head(8).to_string(index=False))

print("\n=== AAR2_-_34824434.23-P1P2 family in S2 ===")
sub2 = s2[s2["family"] == "AAR2_-_34824434.23-P1P2"]
print(sub2[cols].head(8).to_string(index=False))

# ---- genome window for the same families (from S8)
win_map = {}
for fam, grp in s8.groupby("perfect match sgRNA"):
    w = grp["genome input"].dropna()
    if not w.empty:
        win_map[fam] = w.value_counts().index[0]

for fam in ["AARS2_+_44281027.23-P1P2", "AAR2_-_34824434.23-P1P2"]:
    w = win_map.get(fam)
    print(f"\nwindow[{fam}] = {w}")
    if w:
        print(f"   flank   = {w[0:2]}")
        print(f"   guide   = {w[2:22]}")
        print(f"   pam     = {w[22:25]}")
        print(f"   tail    = {w[25:]}")

# ---- does the genome window's guide region equal the family's _00 sequence?
s2_00 = s2[s2["guide_label"].str.endswith("_00")].set_index("guide_label")["sequence"].to_dict()

print("\n=== reference guide: genome window vs S2 _00 sequence ===")
for fam in list(win_map)[:12]:
    w = win_map[fam]
    ref_from_window = w[2:22]
    ref_from_s2 = s2_00.get(fam + "_00")
    same = (ref_from_window == ref_from_s2) if ref_from_s2 else None
    print(f"{fam:42s} window={ref_from_window}  s2_00={ref_from_s2}  same={same}")

# ---- global: compare my computed mismatch set against S2 reported set
GUIDE_LEN = 20
def scan(g, r):
    return [i for i, (a, b) in enumerate(zip(str(g).upper(), str(r).upper()))
            if a != b and a != "N" and b != "N"]

rows = []
for r in s2.to_dict("records"):
    fam = r["family"]
    w = win_map.get(fam)
    if not w or not isinstance(r.get("sequence"), str):
        continue
    ref = w[2:22]
    idxs = scan(r["sequence"], ref)
    from_pam = sorted(-(GUIDE_LEN - i) for i in idxs)
    rep = []
    for k in ("mismatch 1 position", "mismatch 2 position"):
        v = r.get(k)
        if isinstance(v, str):
            try:
                rep.append(int(float(v)))
            except Exception:
                pass
    rep = sorted(rep)
    # S8 reported position (single)
    rows.append(dict(guide=r["guide_label"], n_win=len(idxs),
                     idx0=tuple(idxs), from_pam=tuple(from_pam), reported=tuple(rep),
                     exact=(tuple(from_pam) == tuple(rep)),
                     subset=(set(from_pam) <= set(rep)) if rep else None))

d = pd.DataFrame(rows)
print(f"\n=== global comparison over {len(d)} S2 guides with a window ===")
print("n window diffs:", d["n_win"].value_counts().sort_index().to_dict())
print("exact agreement with S2 reported set:", d["exact"].mean().round(4))
print("my set is subset of reported      :", d["subset"].mean().round(4))

# where do my extra mismatches sit?
d["extra"] = d.apply(lambda r: tuple(sorted(set(r["from_pam"]) - set(r["reported"]))), axis=1)
d["missing"] = d.apply(lambda r: tuple(sorted(set(r["reported"]) - set(r["from_pam"]))), axis=1)
print("\nmost common EXTRA positions I call:", d["extra"].value_counts().head(6).to_dict())
print("most common MISSING positions     :", d["missing"].value_counts().head(6).to_dict())

# exactly-one-diff-of-mine cases: does -20 dominate when n_win==2?
dd = d[d["n_win"] == 2]
print("\namong n_win==2 rows, does my set contain -20?",
      dd["from_pam"].apply(lambda t: -20 in t).mean().round(4))

# restrict to single-diff rows and re-check agreement
d1 = d[d["n_win"] == 1]
print("\nsingle-diff rows:", len(d1), " exact agreement:", d1["exact"].mean().round(4))
