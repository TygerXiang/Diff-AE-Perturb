#!/usr/bin/env python
"""
Fill the remaining sequence gaps from the UCSC sequence API.

Genome builds (important!)
--------------------------
GSE132080 (Dixit et al. 2016) guide labels use **hg19** coordinates.
Verified: RAN_+_131356438 resolves on hg19 chr12 (it does not on hg38),
while RPS15_-_1438413 resolves on both builds with the same PAM (AGG).
GSE283614 (2025) uses **hg38**.

Targets
-------
1) GSE132080: 10 guides in the RAN / RPS15 families lack a PAM because Table S8
   does not cover those families.  The reference protospacer is known from
   Table S2, so only the PAM is missing.
2) GSE283614: 41 targeting guides carry an hg38 coordinate but no target
   sequence / PAM.

Protospacers are located by searching the fetched window on either strand, so
no strand assumption is needed.
"""

import json
import time
import urllib.request
import pandas as pd
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
CACHE = WORKSPACE / "results/.ucsc_cache"
OUT = WORKSPACE / "results/ucsc_fetched_sequences.csv"

API = ("https://api.genome.ucsc.edu/getData/sequence"
       "?genome={genome};chrom={chrom};start={start};end={end}")

TRANS = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def revcomp(s):
    return str(s).translate(TRANS)[::-1]

GSE132080_BUILD = "hg19"
GSE283614_BUILD = "hg38"

GSE132080_RESIDUAL = [
    ("RAN_+_131356438.23-P1P2", "GGCGGTCGCTGCGCTTAGGG", 131356438, None),
    ("RPS15_-_1438413.23-P1P2", "GACCAAAGCGATCTCTTCTG", 1438413, "chr19"),
]
CANDIDATE_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
HALF = 80


def fetch(genome, chrom, start, end):
    """Fetch sequence, caching to disk. start/end are 0-based half-open."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"{genome}_{chrom}_{start}_{end}.json"
    if key.exists():
        return json.loads(key.read_text())["dna"]
    url = API.format(genome=genome, chrom=chrom, start=start, end=end)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as fh:
                payload = json.load(fh)
            key.write_text(json.dumps(payload))
            time.sleep(0.12)
            return payload["dna"]
        except Exception as e:
            if attempt == 2:
                print(f"    fetch failed {genome} {chrom}:{start}-{end}: {e}")
                return None
            time.sleep(1.0)


def locate(genome, chrom, coord, protospacer):
    """Search both strands near coord (1-based). Return dict or None."""
    start = max(0, coord - 1 - HALF)
    end = coord + HALF
    dna = fetch(genome, chrom, start, end)
    if not dna:
        return None
    dna_u = dna.upper()
    p = str(protospacer).upper()

    for seq, strand in ((p, "+"), (revcomp(p), "-")):
        i = dna_u.find(seq)
        if i < 0:
            continue
        ps_start = start + i + 1
        ps_end = ps_start + len(p) - 1
        if strand == "+":
            pam = dna_u[ps_end - start: ps_end - start + 3]
        else:
            pam = revcomp(dna_u[ps_start - start - 4: ps_start - start - 1])
        return dict(genome=genome, chrom=chrom, coord=coord, strand=strand,
                    protospacer_start=ps_start, protospacer_end=ps_end,
                    target_genomic_sequence=p, pam_sequence=pam,
                    pam_valid=bool(len(pam) == 3 and pam[1:] == "GG"),
                    offset_from_label=ps_end - coord)
    return None


def resolve_chrom(genome, protospacer, coord):
    for chrom in CANDIDATE_CHROMS:
        hit = locate(genome, chrom, coord, protospacer)
        if hit:
            return hit
    return None


def main():
    rows = []

    print(f"### GSE132080 residual families (build={GSE132080_BUILD}) -- need PAM only")
    for family, proto, coord, chrom in GSE132080_RESIDUAL:
        if chrom:
            hit = locate(GSE132080_BUILD, chrom, coord, proto)
        else:
            print(f"  resolving chromosome for {family} ...")
            hit = resolve_chrom(GSE132080_BUILD, proto, coord)
        print(f"  {family}: PAM={hit['pam_sequence'] if hit else None} "
              f"strand={hit['strand'] if hit else None}")
        if hit:
            rows.append(dict(source="GSE132080_residual", guide_key=family, **hit))

    print(f"\n### GSE283614 targeting guides (build={GSE283614_BUILD})")
    act = pd.read_csv(
        WORKSPACE / "reproducibility/prepared/gse283614/gse283614_table_s1_activity.tsv",
        sep="\t")
    targets = act[act["starting_position_hg38"].notna() &
                  act["sequence"].notna()].drop_duplicates("guide_identity")
    print(f"  {len(targets)} guides with coordinate + sequence")

    for r in targets.to_dict("records"):
        pos = str(r["starting_position_hg38"]).split("/")[0]
        if ":" not in pos:
            continue
        chrom, coord = pos.split(":")
        try:
            coord = int(float(coord))
        except ValueError:
            continue
        hit = locate(GSE283614_BUILD, chrom, coord, r["sequence"])
        if hit:
            rows.append(dict(source="GSE283614", guide_key=r["guide_identity"], **hit))
        else:
            print(f"  {r['guide_identity']}: NOT FOUND at {chrom}:{coord}")
            rows.append(dict(source="GSE283614", guide_key=r["guide_identity"],
                             genome=GSE283614_BUILD, chrom=chrom, coord=coord,
                             strand=None, protospacer_start=None, protospacer_end=None,
                             target_genomic_sequence=None, pam_sequence=None,
                             pam_valid=None, offset_from_label=None))

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 70)
    print(f"fetched rows : {len(df)}")
    print(f"  resolved   : {df['target_genomic_sequence'].notna().sum()}")
    print(f"  PAM valid  : {(df['pam_valid'] == True).sum()}")
    print(df.groupby("source")["pam_valid"].agg(["count", "sum"]).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
