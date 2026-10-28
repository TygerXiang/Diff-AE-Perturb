#!/usr/bin/env python3
"""Train one Diff-AE-Perturb arm with the activity-path target-id sizing fixed.

The upstream checkout is mounted read-only in the benchmark container, so the fix is installed at
runtime (see tools/patch_perturbdiff_target_ids.py).  The shim must be in place before
``train_with_split`` imports the data/model modules, hence this separate entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORK = Path("/root/perturbench")
sys.path.insert(0, str(WORK / "tools"))

import patch_perturbdiff_target_ids as shim  # noqa: E402
from perturbdiff.data import CONTROL_PATTERNS, encode_perturbation_labels  # noqa: E402

REPO = Path("/root/Diff-AE-Perturb")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def _declare_target_vocab_size(argv: list[str]) -> int:
    """Derive the doser vocabulary from the actual training data and declare it to the shim.

    The doser is indexed by the *target* id space.  With ``--perturbation-col perturbation_label``
    the perturbation labels are already target genes, so both id spaces coincide, but the sizing is
    still derived from the data rather than assumed.
    """
    import anndata as ad

    pert_col = "perturbation_label"
    if "--perturbation-col" in argv:
        pert_col = argv[argv.index("--perturbation-col") + 1]
    paths = [argv[argv.index(flag) + 1] for flag in ("--train-path", "--test-path") if flag in argv]
    best = 0
    for path in paths:
        adata = ad.read_h5ad(path)
        labels = [str(value) for value in adata.obs[pert_col].to_list()]
        _, mapping = encode_perturbation_labels(
            labels,
            control_labels=CONTROL_PATTERNS,
            target_strategy="first_token",
            existing_mapping=None,
        )
        best = max(best, max(int(v) for v in mapping.values()) + 1)
    return best


def main() -> int:
    if not shim.PATCH_APPLIED:
        raise SystemExit("target-id patch failed to install")

    argv = sys.argv[1:]
    target_vocab = _declare_target_vocab_size(argv)
    shim.set_target_vocab_size(target_vocab)
    pert_col = argv[argv.index("--perturbation-col") + 1] if "--perturbation-col" in argv else "default"
    print(
        f"[wrapper] target-id doser fix active; perturbation_col={pert_col} "
        f"target vocab size = {target_vocab}",
        flush=True,
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_with_split", REPO / "scripts" / "train_with_split.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_with_split"] = module
    spec.loader.exec_module(module)
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
