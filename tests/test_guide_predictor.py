from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff.guide_activity import ActivityPrediction, GuideInput
from perturbdiff.guide_predictor import (
    CallableGuideActivityPredictor,
    GuideActivityPredictor,
    TorchFeaturePredictorAdapter,
    make_guide_input,
    predict_activity_tensor,
)


class GuidePredictorAdapterTest(unittest.TestCase):
    def test_protocol_and_input_builder(self) -> None:
        record = make_guide_input("A" * 20, pam="TGG", tss_distance=10)
        predictor = CallableGuideActivityPredictor(lambda guide: 0.75, source="test")
        self.assertIsInstance(predictor, GuideActivityPredictor)
        self.assertEqual(predictor.predict(record).normalized_score, 0.75)
        updated = make_guide_input(record, tss_distance=50)
        self.assertEqual(updated.tss_distance, 50)

    def test_mapping_output_preserves_metadata(self) -> None:
        predictor = CallableGuideActivityPredictor(
            lambda guide: {
                "score": 2.0,
                "normalized_score": 0.4,
                "context_mode": "genome_context",
                "missing_features": ["chromatin_features"],
            },
            source="external",
        )
        result = predictor.predict(GuideInput("A" * 20))
        self.assertEqual(result.source, "external")
        self.assertEqual(result.context_mode, "genome_context")
        self.assertEqual(result.missing_features, ("chromatin_features",))
        self.assertEqual(result.score, 2.0)
        self.assertEqual(result.normalized_score, 0.4)

    def test_torch_adapter_is_frozen_and_batch_tensor_keeps_order(self) -> None:
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(2.0)
        adapter = TorchFeaturePredictorAdapter(
            module,
            feature_builder=lambda guide: [float(len(guide.sequence))],
            source="torch_test",
        )
        values, metadata = predict_activity_tensor(
            adapter,
            [GuideInput("A" * 5), GuideInput("A" * 10)],
        )
        self.assertTrue(torch.allclose(values, torch.tensor([10.0, 20.0])))
        self.assertEqual([item.score for item in metadata], [10.0, 20.0])
        self.assertFalse(module.training)


if __name__ == "__main__":
    unittest.main()
