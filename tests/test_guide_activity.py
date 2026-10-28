from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff.guide_activity import (
    CRISPRiRuleSetPredictor,
    FrozenActivityCalibrator,
    GuideInput,
)


class GuideActivityTest(unittest.TestCase):
    def test_sequence_only_prediction_has_provenance(self) -> None:
        result = CRISPRiRuleSetPredictor().predict(GuideInput("ACGT" * 5))
        self.assertEqual(result.context_mode, "sequence_only")
        self.assertIn("tss_distance", result.missing_features)
        self.assertGreaterEqual(result.normalized_score, 0.0)
        self.assertLessEqual(result.normalized_score, 1.0)

    def test_tss_context_and_pam_change_score(self) -> None:
        predictor = CRISPRiRuleSetPredictor()
        near = predictor.predict(GuideInput("GCGT" * 5, pam="TGG", tss_distance=20))
        far = predictor.predict(GuideInput("GCGT" * 5, pam="TAA", tss_distance=1000))
        self.assertEqual(near.context_mode, "genome_context")
        self.assertGreater(near.score, far.score)

    def test_calibrator_is_monotonic_and_detaches(self) -> None:
        calibrator = FrozenActivityCalibrator()
        x = torch.tensor([0.1, 0.8], requires_grad=True)
        y = calibrator(x)
        self.assertLess(float(y[0]), float(y[1]))
        self.assertEqual(float(calibrator(torch.tensor([0.0]))[0]), 0.0)
        y.sum().backward()
        self.assertIsNone(x.grad)
        self.assertIsNotNone(calibrator.bias.grad)


if __name__ == "__main__":
    unittest.main()
