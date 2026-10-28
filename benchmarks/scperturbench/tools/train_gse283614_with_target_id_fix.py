"""Entry point that installs the target-id sizing fix before importing the repo.

``PerturbDiffusionModel`` sizes the **target-specific** activity doser from
``config.n_perturbations``, which is the *perturbation* vocabulary.  When the two
vocabularies differ the doser is too small and ``target_ids`` indexing raises
``IndexError`` (see reports/diffae_activity_ablation_gse132080.md section 6.1).

For the GSE283614 ablation the perturbation column is ``guide_family``
(``control``/``NANOG``/``OCT4``, vocabulary 3) and the target vocabulary is also
3, so the upstream sizing happens to be sufficient and ``patch()`` is a no-op
(``_TARGET_VOCAB_SIZE <= default_targets``).  The shim is installed anyway so
**both arms run the same entry point**: an A/B where one arm takes a different
code path is not an A/B.

The patch implementation is reused verbatim from
``.scperturbench/work/tools/patch_perturbdiff_target_ids.py`` (the file the
GSE132080 ablation used); this module only guarantees it is importable and
applied before ``train_with_split`` builds a model.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

WORK = Path("/root/perturbench")
REPO = Path("/root/Diff-AE-Perturb")

for candidate in (WORK / ".scperturbench" / "work" / "tools", WORK, REPO / "src"):
    location = str(candidate)
    if candidate.exists() and location not in sys.path:
        sys.path.insert(0, location)

import patch_perturbdiff_target_ids as _patch  # noqa: E402

_patch.patch()

if __name__ == "__main__":
    entry = REPO / "scripts" / "train_with_split.py"
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name="__main__")
