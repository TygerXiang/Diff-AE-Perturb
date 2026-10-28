"""Diagnose why held-out guide conditions are reported as unseen by the mapping.

Reads a trained checkpoint's ``perturbation_mapping`` and the fold's split files,
then shows, for every held-out guide, exactly which candidate labels the adapter
will try and whether each one resolves.  Read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import torch

REPO_SRC = Path("/root/Diff-AE-Perturb/src")
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from perturbdiff.data import canonical_guide_keys, extract_target_label, is_control_label  # noqa: E402


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "esc"
    fold = sys.argv[2] if len(sys.argv) > 2 else "0"
    arm = sys.argv[3] if len(sys.argv) > 3 else "activity_on"
    split = Path("/root/Diff-AE-Perturb/reproducibility/prepared/gse283614/dose_cv") / prefix / f"fold_{fold}"
    ckpt_dir = Path("/root/perturbench/.scperturbench/trained/gse283614_activity_ablation") / prefix / f"fold_{fold}" / arm
    ckpt = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))[-1]

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    mapping = state["perturbation_mapping"]
    print("checkpoint:", ckpt.name)
    print("perturbation_mapping:", mapping)
    cfg = state["model_config"]
    print("n_perturbations(config):", cfg.n_perturbations)
    print("guide_encoder.enabled:", cfg.guide_encoder.enabled,
          "activity_source:", cfg.guide_encoder.activity_source,
          "feature_dim:", cfg.guide_encoder.feature_dim,
          "explicit_condition:", cfg.guide_encoder.explicit_condition)

    train = ad.read_h5ad(split / f"gse283614_{prefix}_train.h5ad", backed="r")
    test = ad.read_h5ad(split / f"gse283614_{prefix}_test.h5ad", backed="r")
    for name, adata in (("train", train), ("test", test)):
        for col in ("perturbation_label", "guide_family", "guide_identity"):
            if col in adata.obs:
                vc = adata.obs[col].astype(str).value_counts()
                print(f"{name}.{col}: n_levels={len(vc)} -> {dict(list(vc.items())[:6])}")

    control = "control"
    for eval_level in ("target", "guide"):
        unseen = []
        for guide in sorted(set(test.obs["perturbation_label"].astype(str)) - {control}):
            target_label = guide  # what scperturbench_adapters.adapter_diffae passes
            candidates = [guide, target_label] if eval_level == "guide" else [target_label, guide]
            resolved = None
            for candidate in candidates:
                if resolved is not None:
                    break
                hit = mapping.get(str(candidate))
                if hit is not None:
                    resolved = (candidate, hit)
                    break
                fallback = extract_target_label(str(candidate), strategy="first_token")
                hit = mapping.get(fallback)
                if hit is not None:
                    resolved = (fallback, hit)
                    break
            if resolved is None:
                unseen.append(guide)
        print(f"eval_level={eval_level}: unresolved={unseen}")

    print("is_control_label('control') =", is_control_label("control"))
    print("is_control_label('NANOG')  =", is_control_label("NANOG"))
    print("canonical_guide_keys('NANOG-1-5') =", canonical_guide_keys("NANOG-1-5"))
    print("canonical_guide_keys('NANOG-pro1') =", canonical_guide_keys("NANOG-pro1"))
    print("json dump mapping:", json.dumps(mapping))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
