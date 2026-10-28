"""Fix the activity-path target-id sizing for guide-level Diff-AE-Perturb training.

The bug (two coupled defects, both in the read-only upstream checkout)
--------------------------------------------------------------------
For a guide-level split (``--perturbation-col guide_identity``) the data loader builds two distinct
id spaces:

    perturbation_ids -> guide vocabulary      (103 entries: 102 real guides + control)
    target_ids       -> target-gene vocabulary (128 entries: 127 target genes + control)

``target_ids`` is not indexed into the perturbation embedding, however: it indexes the
**target-specific** ``GeneralizedSigmoid`` doser (``self.bias[0][target_ids]``), whose tables are
sized ``n_targets + 1``.  ``PerturbDiffusionModel`` builds that doser from
``config.n_perturbations``, i.e. the *guide* vocabulary size, so any cell whose target id exceeds the
guide-vocabulary size fails with::

    IndexError: index 104 is out of bounds for dimension 0 with size 104

Because the no-activity arm never evaluates ``target_ids``, the defect is invisible there and the
A/B comparison would otherwise compare a working arm against a broken one.

The fix
-------
Keep both id spaces as the data loader defines them (a fresh target vocabulary is semantically
correct: the doser is target-specific) and size the doser to the actual target vocabulary.  The
shim wraps ``PerturbDiffusionModel.__init__``: it lets the model build with the default sizing and
then re-creates ``guide_doser`` with the corrected ``n_targets``, preserving the parameter shapes
that are consistent with it.  ``config.n_perturbations`` still governs the perturbation embedding,
which must stay guide-indexed.

Apply before importing anything that builds a model::

    import patch_perturbdiff_target_ids  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_SRC = Path("/root/Diff-AE-Perturb/src")


def _ensure_repo_on_path() -> None:
    location = str(REPO_SRC)
    if location not in sys.path:
        sys.path.insert(0, location)


_ensure_repo_on_path()

import torch  # noqa: E402
from torch import nn  # noqa: E402

from perturbdiff import model as _model  # noqa: E402
from perturbdiff.modules import GeneralizedSigmoid  # noqa: E402

PATCH_APPLIED = False

# Set from the training data before the model is constructed; None keeps upstream behaviour.
_TARGET_VOCAB_SIZE: int | None = None


def set_target_vocab_size(size: int | None) -> None:
    """Declare how many distinct target ids the doser must be able to index."""
    global _TARGET_VOCAB_SIZE
    _TARGET_VOCAB_SIZE = None if size is None else int(size)


def get_target_vocab_size() -> int | None:
    return _TARGET_VOCAB_SIZE


if not hasattr(_model.PerturbDiffusionModel, "_original_init"):
    _model.PerturbDiffusionModel._original_init = _model.PerturbDiffusionModel.__init__


def _patched_init(self, config):  # type: ignore[no-untyped-def]
    _model.PerturbDiffusionModel._original_init(self, config)
    if _TARGET_VOCAB_SIZE is None or getattr(self, "guide_doser", None) is None:
        return
    default_targets = int(config.n_perturbations)
    if _TARGET_VOCAB_SIZE <= default_targets:
        return  # upstream sizing is already sufficient
    # Rebuild the doser so target-id indexing is in range, keeping the configured nonlinearity.
    doser = GeneralizedSigmoid(
        n_targets=_TARGET_VOCAB_SIZE,
        non_linearity=getattr(self.guide_doser, "non_linearity", "sigm"),
    )
    with torch.no_grad():
        # Preserve the neutral initialisation (beta=1, bias=0) rather than copying the smaller table,
        # which would leave the extra target rows at an inconsistent scale.
        doser.beta.fill_(1.0)
        doser.bias.zero_()
    doser = doser.to(next(self.parameters()).device)
    self.guide_doser = doser
    self._target_vocab_size_override = _TARGET_VOCAB_SIZE


def patch() -> bool:
    global PATCH_APPLIED
    _model.PerturbDiffusionModel.__init__ = _patched_init
    PATCH_APPLIED = True
    return True


def describe() -> dict[str, object]:
    return {
        "patch_applied": PATCH_APPLIED,
        "target_vocab_size": _TARGET_VOCAB_SIZE,
    }


patch()
