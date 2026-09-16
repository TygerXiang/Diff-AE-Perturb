#!/usr/bin/env python
"""Investigate sequence alignment conventions in GSE132080 Table S8/S2."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\Admin\myprojects\diff-project\datasets\GSE132080\41587_2019_387_MOESM3_ESM")
S8 = BASE / "Table_S8_machine_learning_input.txt"

TRANS = str.maketrans('ACGTN', 'TGCAN')
def revcomp(s):
    return str(s).translate(TRANS)[::-1]

def load_s8():
    df = pd.read_csv(S8, sep='\t', dtype=str)
    df = df.rename(columns={df.columns[0]: 'guide'})
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df

df = load_s8()
print("S8 shape:", df.shape)
print("columns:", list(df.columns))

gl = df['genome input'].str.len()
sl = df['sgRNA input'].str.len()
ql = df['sgRNA sequence'].str.len()
print("\ngenome input lengths:", gl.value_counts().to_dict())
print("sgRNA input lengths:", sl.value_counts().to_dict())
print("sgRNA sequence lengths:", ql.value_counts().to_dict())

df['strand'] = df['guide'].str.extract(r'_([+-])_')
print("\nstrand counts:", df['strand'].value_counts().to_dict())

def off_direct(row):
    return row['sgRNA input'].find(row['sgRNA sequence'])
def off_rc(row):
    return row['sgRNA input'].find(revcomp(row['sgRNA sequence']))

df['off_direct'] = [off_direct(r) for r in df.to_dict('records')]
df['off_rc'] = [off_rc(r) for r in df.to_dict('records')]

print("\noffset of guide(variant) inside sgRNA input -- DIRECT:")
print(pd.crosstab(df['strand'], df['off_direct']).to_string())
print("\noffset of guide(variant) inside sgRNA input -- REVCOMP:")
print(pd.crosstab(df['strand'], df['off_rc']).to_string())

def diffpos(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]

df['win_diffs'] = [diffpos(r['genome input'], r['sgRNA input']) for r in df.to_dict('records')]
df['n_win_diffs'] = df['win_diffs'].apply(len)
print("\nnumber of diffs between genome input and sgRNA input:")
print(df['n_win_diffs'].value_counts().sort_index().to_dict())

# relate window diff position to the reported mismatch position
print("\n=== examples ===")
for strand in ['+', '-']:
    sub = df[df['strand'] == strand].head(3)
    for _, r in sub.iterrows():
        print(f"\nstrand={strand}  guide={r['guide']}")
        print(f"  perfect match : {r['perfect match sgRNA']}")
        print(f"  genome input  : {r['genome input']}  (len {len(r['genome input'])})")
        print(f"  sgRNA  input  : {r['sgRNA input']}  (len {len(r['sgRNA input'])})")
        print(f"  sgRNA sequence: {r['sgRNA sequence']}  (len {len(r['sgRNA sequence'])})")
        print(f"  mismatch pos  : {r['mismatch position']}   new pairing: {r['new pairing']}")
        print(f"  window diffs  : {r['win_diffs']}")
        print(f"  off_direct={r['off_direct']}  off_rc={r['off_rc']}")

# Check: does guide sit at window[2:22]? where is PAM?
print("\n=== PAM hypothesis check ===")
for off_choice, name in [(2, 'guide at [2:22]'), (3, 'guide at [3:23]')]:
    ok = 0; tot = 0
    for r in df.head(3000).to_dict('records'):
        w = r['sgRNA input']
        if len(w) < off_choice + 23:
            continue
        pam = w[off_choice + 20: off_choice + 23]
        tot += 1
        if len(pam) == 3 and pam[1] == 'G' and pam[2] == 'G':
            ok += 1
    print(f"{name}: PAM ends with GG in {ok}/{tot} = {100*ok/max(tot,1):.1f}%")

# Also test the genome-input version
for off_choice in [2, 3]:
    ok = 0; tot = 0
    for r in df.head(3000).to_dict('records'):
        w = r['genome input']
        if len(w) < off_choice + 23:
            continue
        pam = w[off_choice + 20: off_choice + 23]
        tot += 1
        if len(pam) == 3 and pam[1] == 'G' and pam[2] == 'G':
            ok += 1
    print(f"genome input, guide at [{off_choice}:{off_choice+20}]: PAM GG in {ok}/{tot} = {100*ok/max(tot,1):.1f}%")

# check the relationship mismatch_position -> window diff, using guide offset
print("\n=== mismatch position -> distance to PAM ===")
r = df.iloc[0]
print("example mismatch position:", r['mismatch position'])
