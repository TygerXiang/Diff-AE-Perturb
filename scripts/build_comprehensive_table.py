#!/usr/bin/env python
"""
Build the complete, validated sgRNA-level analysis table.

Sequence conventions (established empirically, see docs/):
  * Table S2 `_00` row of each family = the reference protospacer (genomic
    sequence, already oriented in the guide's own direction).
    Using it as the reference reproduces the published mismatch annotations
    for 117,010 / 117,010 guides EXACTLY (agreement = 1.000).
  * Table S8 `genome input` = 26 bp window, guide-oriented:
        [0:2]   upstream flank
        [2:22]  protospacer as read from the genome
        [22:25] PAM (100% end in GG)
        [25]    one extra downstream base
    NOTE: base 0 of the S8 window protospacer disagrees with the library
    reference for a subset of families, so the S8 window is used ONLY as the
    source of the PAM, never for mismatch calling.
  * Published mismatch position -N  <->  guide index (20 - N)
                                    <->  distance to PAM (N - 1)
"""

import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
GSE132080_DIR = WORKSPACE / "datasets" / "GSE132080" / "41587_2019_387_MOESM3_ESM"
S8_PATH = GSE132080_DIR / "Table_S8_machine_learning_input.txt"
S2_PATH = GSE132080_DIR / "Table_S2_large-scale_library_sequences_revised.txt"

GUIDE_SCALES = WORKSPACE / "runs/gse132080_sgrna_200ep/guide_scale_export/guide_virtual_scales.csv"
GSE283614_GUIDES = WORKSPACE / "reproducibility/prepared/gse283614/gse283614_de_guide_assignments.csv"
GSE283614_ACTIVITY = WORKSPACE / "reproducibility/prepared/gse283614/gse283614_table_s1_activity.tsv"
FETCHED = WORKSPACE / "results/ucsc_fetched_sequences.csv"

FLANK, GLEN, PAM = 2, 20, 3
LABEL_RE = re.compile(r"^(?P<gene>.+?)_(?P<strand>[+-])_(?P<coord>\d+)\.")


def read_guide_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={df.columns[0]: "guide_label"}).dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.duplicated()]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return df


def normalize_label(label):
    if pd.isna(label) or label in ("control", "*"):
        return label
    p = str(label).split("_")
    return "_".join(p[1:]) if len(p) >= 2 and p[0] == p[1] else str(label)


def family_of(norm):
    if pd.isna(norm) or norm in ("control", "*"):
        return None
    return str(norm).rsplit("_", 1)[0]


def gc_content(seq):
    if not isinstance(seq, str) or not seq:
        return np.nan
    return round(sum(b in "GC" for b in seq.upper()) / len(seq), 4)


def mismatch_scan(guide, ref):
    if not isinstance(guide, str) or not isinstance(ref, str):
        return []
    return [(i, r, g) for i, (g, r) in enumerate(zip(guide.upper(), ref.upper()))
            if g != r and g != "N" and r != "N"]


# --------------------------------------------------------------------------- #
def build_gse132080(scales, s8, s2):
    # --- reference protospacer per family, from the S2 _00 row -------------
    ref_map = (s2[s2["guide_label"].astype(str).str.endswith("_00")]
               .drop_duplicates("perfect match sgRNA")
               .set_index("perfect match sgRNA")["sequence"].to_dict())
    print(f"    reference protospacers (S2 _00): {len(ref_map)} families")

    # --- PAM from the S8 genome window ------------------------------------
    pam_map, win_map = {}, {}
    for fam, grp in s8.groupby("perfect match sgRNA"):
        w = grp["genome input"].dropna()
        if w.empty:
            continue
        win = w.value_counts().index[0]
        win_map[fam] = win
        pam_map[fam] = win[FLANK + GLEN: FLANK + GLEN + PAM]
    print(f"    genome windows with PAM (S8)   : {len(pam_map)} families")

    s2_cols = ["guide_label", "perfect match sgRNA", "sequence",
               "mismatch 1 position", "new pairing mismatch 1",
               "mismatch 2 position", "new pairing mismatch 2",
               "off-target threshold (genome)",
               "off-target specificity score (CFD)"]
    s2_sub = s2[[c for c in s2_cols if c in s2.columns]].drop_duplicates("guide_label")
    s8_gamma = (s8[["guide_label", "mean relative gamma", "K562", "Jurkat"]]
                .drop_duplicates("guide_label")
                .rename(columns={"mean relative gamma": "mean_relative_gamma"}))

    df = scales.copy()
    df["guide_label_normalized"] = df["label"].apply(normalize_label)
    df = df.merge(s2_sub, left_on="guide_label_normalized", right_on="guide_label",
                  how="left", suffixes=("", "_s2"))
    df = df.merge(s8_gamma, on="guide_label", how="left")

    recs = []
    for r in df.to_dict("records"):
        norm = r.get("guide_label_normalized")
        family = r.get("perfect match sgRNA") or family_of(norm)
        guide = r.get("sequence") or r.get("sequence_s2")
        ref = ref_map.get(family)
        pam = pam_map.get(family)
        win = win_map.get(family)

        m = LABEL_RE.match(str(norm)) if norm else None
        rec = {
            "family_label": family,
            "guide_label_normalized": norm,
            "target_gene_label": m.group("gene") if m else None,
            "strand": m.group("strand") if m else None,
            "genomic_coordinate": int(m.group("coord")) if m else np.nan,
            "sgrna_sequence": guide,
            "target_genomic_sequence": ref,
            "reference_source": "TableS2_family_00" if ref else None,
            "pam_sequence": pam,
            "pam_inferred": "NGG" if (pam is None and ref) else None,
            "pam_source": "TableS8_genome_window" if pam else ("not_available" if ref else None),
            "genome_window_26bp": win,
            "genome_window_protospacer": win[FLANK:FLANK + GLEN] if win else None,
            "upstream_flank": win[:FLANK] if win else None,
            "window_pos0_discrepancy": (bool(win) and bool(ref)
                                        and win[FLANK:FLANK + GLEN] != ref),
            "n_mismatches": np.nan,
            "mismatch_positions_from_5p": None,
            "mismatch_positions_from_pam": None,
            "mismatch_distance_to_pam": None,
            "min_distance_to_pam": np.nan,
            "mismatch_ref_bases": None,
            "mismatch_alt_bases": None,
            "mismatch_validated_vs_table_s2": None,
            "s2_mismatch_1_position": r.get("mismatch 1 position"),
            "s2_mismatch_2_position": r.get("mismatch 2 position"),
            "s2_new_pairing_1": r.get("new pairing mismatch 1"),
            "s2_new_pairing_2": r.get("new pairing mismatch 2"),
            "cfd_specificity_score": r.get("off-target specificity score (CFD)"),
            "off_target_threshold_genome": r.get("off-target threshold (genome)"),
        }

        hits = mismatch_scan(guide, ref)
        if ref:
            rec["n_mismatches"] = len(hits)
            if hits:
                idx = [h[0] for h in hits]
                from_pam = [-(GLEN - i) for i in idx]
                dist = [(GLEN - 1) - i for i in idx]
                rec["mismatch_positions_from_5p"] = ",".join(map(str, idx))
                rec["mismatch_positions_from_pam"] = ",".join(map(str, from_pam))
                rec["mismatch_distance_to_pam"] = ",".join(map(str, dist))
                rec["min_distance_to_pam"] = int(min(dist))
                rec["mismatch_ref_bases"] = ",".join(h[1] for h in hits)
                rec["mismatch_alt_bases"] = ",".join(h[2] for h in hits)

            reported = []
            for v in (rec["s2_mismatch_1_position"], rec["s2_mismatch_2_position"]):
                if isinstance(v, str):
                    try:
                        reported.append(int(float(v)))
                    except Exception:
                        pass
            if reported:
                rec["mismatch_validated_vs_table_s2"] = (
                    sorted(reported) == sorted(from_pam) if hits else False)
        recs.append(rec)

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out["dataset"] = "GSE132080"
    return out


def build_gse283614(guide_file, activity_file):
    guides = pd.read_csv(guide_file)
    act = pd.read_csv(activity_file, sep="\t")
    uniq = guides[["guide_identity", "target_gene", "guide_type", "guide_family"]].drop_duplicates()
    keep = [c for c in ["guide_identity", "sequence", "starting_position_hg38",
                        "screen_strength_mean", "relative_activity_mean",
                        "screen_strength_ESC", "relative_activity_ESC",
                        "screen_strength_DE", "relative_activity_DE"] if c in act.columns]
    out = uniq.merge(act[keep], on="guide_identity", how="left")

    pos = out["starting_position_hg38"].astype(str) if "starting_position_hg38" in out else pd.Series([None] * len(out))
    out["chromosome"] = pos.str.split(":").str[0].replace({"nan": None})
    out["position"] = pd.to_numeric(pos.str.split(":").str[1].str.split("/").str[0], errors="coerce")
    out = out.rename(columns={"sequence": "sgrna_sequence",
                              "starting_position_hg38": "genomic_position_hg38"})
    if "sgrna_sequence" not in out:
        out["sgrna_sequence"] = np.nan
    out["sgrna_length"] = out["sgrna_sequence"].apply(lambda s: len(s) if isinstance(s, str) else np.nan)
    out["sgrna_gc_content"] = out["sgrna_sequence"].apply(gc_content)
    out["dataset"] = "GSE283614"
    out["reference_source"] = "not_applicable_design_guides"
    for c in ["family_label", "target_gene_label", "strand", "genomic_coordinate",
              "target_genomic_sequence", "pam_sequence", "pam_inferred", "pam_source",
              "genome_window_26bp", "genome_window_protospacer", "upstream_flank",
              "window_pos0_discrepancy", "n_mismatches", "mismatch_positions_from_5p",
              "mismatch_positions_from_pam", "mismatch_distance_to_pam",
              "min_distance_to_pam", "mismatch_ref_bases", "mismatch_alt_bases",
              "mismatch_validated_vs_table_s2", "cfd_specificity_score",
              "off_target_threshold_genome", "mean_relative_gamma"]:
        out[c] = np.nan
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=WORKSPACE / "results/comprehensive_sgrna_table.csv")
    args = ap.parse_args()

    print("Loading inputs ...")
    scales = pd.read_csv(GUIDE_SCALES)
    scales["guide_label_orig"] = scales["label"]
    scales = scales.rename(columns={"target": "target_gene", "s": "virtual_scale",
                                    "embedding_l2_norm": "embedding_norm"})
    s8 = read_guide_tsv(S8_PATH)
    s2 = read_guide_tsv(S2_PATH)
    print(f"  S8 rows={len(s8)}  S2 rows={len(s2)}  model guides={len(scales)}")

    print("\nBuilding GSE132080 ...")
    g1 = build_gse132080(scales, s8, s2)
    g1["sgrna_length"] = g1["sgrna_sequence"].apply(
        lambda s: len(s) if isinstance(s, str) else np.nan)
    g1["sgrna_gc_content"] = g1["sgrna_sequence"].apply(gc_content)
    print("Building GSE283614 ...")
    g2 = build_gse283614(GSE283614_GUIDES, GSE283614_ACTIVITY)

    g1["guide_identity"] = g1["guide_label_orig"]
    g1["guide_label"] = g1["guide_label_orig"]

    g1 = g1.loc[:, ~g1.columns.duplicated()]
    g2 = g2.loc[:, ~g2.columns.duplicated()]
    common = sorted(set(g1.columns) | set(g2.columns))
    full = pd.concat([g1.reindex(columns=common), g2.reindex(columns=common)], ignore_index=True)

    priority = [
        "dataset", "guide_identity", "guide_label", "guide_label_normalized", "family_label",
        "target_gene", "target_gene_label", "guide_type", "strand",
        "sgrna_sequence", "sgrna_length", "sgrna_gc_content",
        "target_genomic_sequence", "pam_sequence", "pam_inferred", "pam_source",
        "reference_source", "upstream_flank", "genome_window_protospacer", "genome_window_26bp",
        "window_pos0_discrepancy",
        "n_mismatches", "mismatch_positions_from_5p", "mismatch_positions_from_pam",
        "mismatch_distance_to_pam", "min_distance_to_pam",
        "mismatch_ref_bases", "mismatch_alt_bases",
        "s2_mismatch_1_position", "s2_mismatch_2_position", "mismatch_validated_vs_table_s2",
        "cfd_specificity_score", "off_target_threshold_genome",
        "genomic_coordinate", "chromosome", "position", "genomic_position_hg38",
        "virtual_scale", "embedding_norm",
        "mean_relative_gamma", "K562", "Jurkat",
        "screen_strength_mean", "relative_activity_mean",
        "screen_strength_ESC", "relative_activity_ESC",
        "screen_strength_DE", "relative_activity_DE", "guide_family",
    ]
    ordered = [c for c in priority if c in full.columns]
    ordered += [c for c in full.columns if c not in ordered]
    full = full[ordered]

    # ---- fold in UCSC-fetched sequences where the publications lack them ----
    if FETCHED.exists():
        fx = pd.read_csv(FETCHED)
        print(f"\nMerging UCSC-fetched sequences from {FETCHED.name} ...")

        fx132 = (fx[fx["source"] == "GSE132080_residual"]
                 .set_index("guide_key"))
        need = (full["dataset"] == "GSE132080") & full["pam_sequence"].isna()
        mapped = full.loc[need, "family_label"].map(fx132["pam_sequence"])
        full.loc[need, "pam_sequence"] = mapped
        filled = need & full["pam_sequence"].notna()
        full.loc[filled, "pam_source"] = "UCSC_" + fx132["genome"].iloc[0]
        full.loc[filled, "pam_inferred"] = None
        print(f"  GSE132080 residual PAM filled: {int(filled.sum())}")

        fx283 = (fx[fx["source"] == "GSE283614"]
                 .set_index("guide_key"))
        m = full["dataset"] == "GSE283614"
        for col in ("target_genomic_sequence", "pam_sequence", "strand"):
            full.loc[m, col] = (full.loc[m, "guide_identity"]
                                .map(fx283[col]).fillna(full.loc[m, col]))
        got = m & full["pam_sequence"].notna()
        full.loc[got, "pam_source"] = "UCSC_" + str(fx283["genome"].iloc[0])
        full.loc[got, "reference_source"] = "UCSC_hg38_protospacer"
        print(f"  GSE283614 target sequence filled: {int(got.sum())}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.output, index=False)

    print("\n" + "=" * 74)
    print("COMPLETE TABLE SUMMARY")
    print("=" * 74)
    print(f"rows={len(full)}  columns={len(full.columns)}")
    print(full["dataset"].value_counts().to_string())

    real = full[~full["guide_identity"].isin(["control", "*"])]
    print(f"\nreal guides: {len(real)}")
    for c in ["sgrna_sequence", "target_genomic_sequence", "pam_sequence",
              "n_mismatches", "min_distance_to_pam", "embedding_norm",
              "virtual_scale", "mean_relative_gamma", "relative_activity_mean"]:
        n = real[c].notna().sum()
        print(f"  {c:26s} {n:4d}/{len(real)}  ({100*n/len(real):5.1f}%)")

    g1r = real[real["dataset"] == "GSE132080"]
    print(f"\nGSE132080 real guides: {len(g1r)}")
    print(f"  reference protospacer : {g1r['target_genomic_sequence'].notna().sum()}")
    print(f"  PAM 3-mer available   : {g1r['pam_sequence'].notna().sum()}")
    print(f"  mismatch annotated    : {g1r['n_mismatches'].notna().sum()}")
    v = g1r["mismatch_validated_vs_table_s2"].dropna()
    print(f"  mismatch == published : {int(v.sum())}/{len(v)}  ({100*v.mean():.1f}%)")
    print("\n  mismatch count distribution:")
    print(g1r["n_mismatches"].value_counts().sort_index().to_string())
    print("\n  PAM composition:")
    print(g1r["pam_sequence"].value_counts().to_string())
    d = g1r["min_distance_to_pam"].dropna()
    print(f"\n  distance-to-PAM mean={d.mean():.2f} median={d.median():.0f} "
          f"range={d.min():.0f}-{d.max():.0f}")
    print(f"\n  S8-window base-0 discrepancy flagged: {g1r['window_pos0_discrepancy'].sum()}")

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
