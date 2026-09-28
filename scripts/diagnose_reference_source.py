#!/usr/bin/env python
"""Decisive test: use the family _00 sequence from Table S2 as the reference protospacer."""
import numpy as np
import pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "datasets" / "GSE132080" / "41587_2019_387_MOESM3_ESM"

def read_tsv(p):
    df = pd.read_csv(p, sep="\t", dtype=str)
    df = df.rename(columns={df.columns[0]: "guide_label"}).dropna(axis=1, how="all")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return df

s8 = read_tsv(D / "Table_S8_machine_learning_input.txt")
s2 = read_tsv(D / "Table_S2_large-scale_library_sequences_revised.txt")
s2["family"] = s2["perfect match sgRNA"]

GUIDE_LEN = 20
def scan(g, r):
    return [i for i, (a, b) in enumerate(zip(str(g).upper(), str(r).upper()))
            if a != b and a != "N" and b != "N"]

win_map = {}
for fam, grp in s8.groupby("perfect match sgRNA"):
    w = grp["genome input"].dropna()
    if not w.empty:
        win_map[fam] = w.value_counts().index[0]

# reference protospacer = family _00 sequence
s2_00 = (s2[s2["guide_label"].str.endswith("_00")]
         .drop_duplicates("family")
         .set_index("family")["sequence"].to_dict())
print(f"families with a _00 reference sequence : {len(s2_00)}")
print(f"families with an S8 genome window     : {len(win_map)}")
print(f"overlap                               : {len(set(s2_00) & set(win_map))}")

rows = []
for r in s2.to_dict("records"):
    fam = r["family"]
    ref = s2_00.get(fam)
    if not isinstance(ref, str) or not isinstance(r.get("sequence"), str):
        continue
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
    rows.append(dict(guide=r["guide_label"], fam=fam, n=len(idxs),
                     mine=tuple(from_pam), reported=tuple(rep),
                     exact=(tuple(from_pam) == tuple(rep))))

d = pd.DataFrame(rows)
print(f"\n=== reference = S2 _00 sequence, over {len(d)} guides ===")
print("exact agreement with published S2 mismatch set:", round(d["exact"].mean(), 4))
print("\nmismatch-count distribution (mine):")
print(d["n"].value_counts().sort_index().to_string())
print("\nagreement by my mismatch count:")
print(d.groupby("n")["exact"].agg(["mean", "size"]).round(4).to_string())

bad = d[~d["exact"]]
print(f"\ndisagreements: {len(bad)}")
if len(bad):
    print(bad.head(10).to_string(index=False))

# ---- now check the 128 guides actually used by the model ------------------
scales = pd.read_csv(r"C:\Users\Admin\myprojects\Diff-AE-Perturb\runs\gse132080_sgrna_200ep\guide_scale_export\guide_virtual_scales.csv")
def norm(l):
    if pd.isna(l) or l in ("control", "*"):
        return l
    p = str(l).split("_")
    return "_".join(p[1:]) if len(p) >= 2 and p[0] == p[1] else str(l)
scales["nl"] = scales["label"].apply(norm)
scales["fam"] = scales["nl"].apply(lambda x: str(x).rsplit("_", 1)[0] if pd.notna(x) and x not in ("control", "*") else None)
real = scales[scales["fam"].notna()]
print(f"\n=== model guides: {len(real)} ===")
print("families with _00 reference :", real["fam"].isin(s2_00).sum())
print("families with S8 window     :", real["fam"].isin(win_map).sum())
print("families with both          :", (real["fam"].isin(s2_00) & real["fam"].isin(win_map)).sum())
print("\nfamilies missing a window:")
print(real.loc[~real["fam"].isin(win_map), ["label", "fam"]].to_string(index=False))
