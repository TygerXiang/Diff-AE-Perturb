from pathlib import Path as _Path

from .config import DiffusionConfig, GuideFeatureEncoderConfig, ModelConfig, TrainingLossConfig
from .data import (
    PerturbationDataset,
    build_virtual_guide_scales_from_embedding,
    create_dataloader,
    encode_perturbation_labels,
)
from .guide_features import GuideStrengthEncoder as GuideFeatureEncoder
from .guide_activity import (
    ActivityPrediction,
    CRISPRiRuleSetPredictor,
    FrozenActivityCalibrator,
    GuideInput,
    load_guide_input_lookup,
    load_guide_score_lookup,
    predict_activity_batch,
)
from .guide_predictor import (
    CallableGuideActivityPredictor,
    GuideActivityPredictor,
    TorchFeaturePredictorAdapter,
    make_guide_input,
    predict_activity_tensor,
)
from .model import PerturbDiffusionModel

# This distribution and `diff-project` both ship an import package called `perturbdiff`, and the
# host environment has an editable install of the latter, so `import perturbdiff` can silently
# resolve to a different, older implementation.  These markers let any entry point assert that it
# loaded the intended one; see scripts/check_perturbdiff_import.py.
__project__ = "Diff-AE-Perturb"
__project_root__ = str(_Path(__file__).resolve().parents[2])

__all__ = [
    "DiffusionConfig",
    "GuideFeatureEncoder",
    "GuideFeatureEncoderConfig",
    "GuideInput",
    "ActivityPrediction",
    "CRISPRiRuleSetPredictor",
    "FrozenActivityCalibrator",
    "load_guide_input_lookup",
    "load_guide_score_lookup",
    "predict_activity_batch",
    "GuideActivityPredictor",
    "CallableGuideActivityPredictor",
    "TorchFeaturePredictorAdapter",
    "make_guide_input",
    "predict_activity_tensor",
    "ModelConfig",
    "PerturbationDataset",
    "PerturbDiffusionModel",
    "TrainingLossConfig",
    "build_virtual_guide_scales_from_embedding",
    "create_dataloader",
    "encode_perturbation_labels",
]
