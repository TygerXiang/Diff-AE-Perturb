"""Quantify the guide-level vs target-level id space collision for the GSE132080 split."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad

REPO = Path("/root/Diff-AE-Perturb")
sys.path.insert(0, str(REPO / "src"))
from perturbdiff.data import CONTROL_PATTERNS, encode_perturbation_labels, extract_target_label  # noqa: E402

SPLITS = Path("/root/perturbench/.scperturbench/inputs/gse132080_squidiff_repro")
GUIDE_COL = "guide_identity"

for name in ("squidiff_train.h5ad", "squidiff_test.h5ad"):
    adata = ad.read_h5ad(SPLITS / name)
    labels = [str(v) for v in adata.obs[GUIDE_COL].to_list()]
    guide_ids, guide_map = encode_perturbation_labels(
        labels, control_labels=CONTROL_PATTERNS, target_strategy="raw", existing_mapping=None
    )
    target_ids, target_map = encode_perturbation_labels(
        labels, control_labels=CONTROL_PATTERNS, target_strategy="first_token", existing_mapping=None
    )
    guide_ids = [int(v) for v in guide_ids]
    target_ids = [int(v) for v in target_ids]
    guide_idx = {v: k for k, v in guide_map.items()}
    target_idx = {v: k for k, v in target_map.items()}
    mismatched = {
        guide_idx[gid]: {"guide_id": gid, "target_id": tid, "target_name": target_idx.get(tid)}
        for gid, tid in zip(guide_ids, target_ids)
        if gid != 0 and gid != tid
    }
    print(f"=== {name} ===")
    print(f"  n_cells             : {len(labels)}")
    print(f"  guide vocab         : {len(guide_map)} (max id {max(guide_map.values())})")
    print(f"  target vocab        : {len(target_map)} (max id {max(target_map.values())})")
    print(f"  cells where id != id: {len(mismatched)} / {len(labels)}")
    print(f"  max target id used  : {max(target_ids)}")
    sample = dict(list(mismatched.items())[:5])
    print(f"  sample mismatches   : {sample}")
    print(f"  target vocab names  : {sorted(target_map)[:8]} ...")
