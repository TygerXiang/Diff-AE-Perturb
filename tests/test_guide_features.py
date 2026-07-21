from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff.guide_features import (
    GuideFeatureRecord,
    GuideStrengthEncoder,
    build_notebook_cnn_feature_tensor,
    canonical_guide_keys,
    guide_cnn_feature_dim,
)


class GuideFeatureTest(unittest.TestCase):
    def test_canonical_keys_match_prefixed_labels(self) -> None:
        keys = canonical_guide_keys("DBR1_DBR1_+_137893744.23-P1P2_00")

        self.assertIn("DBR1_+_137893744.23-P1P2_00", keys)

    def test_notebook_cnn_feature_tensor_shape_is_stable(self) -> None:
        record = GuideFeatureRecord(
            label="AAR2_-_34824434.23-P1P2_01",
            gene="AAR2",
            sequence="GTGAGGCGAGGCGGTGAGTG",
            genome_input="ACGTGGGGCGAGGCGGTGAGTGTGGC",
            sgrna_input="ACGTGAGGCGAGGCGGTGAGTGTGGC",
        )

        feature = build_notebook_cnn_feature_tensor(record, sequence_length=26)

        self.assertEqual(feature.shape, (4, 26, 2))

    def test_guide_strength_encoder_output_shape(self) -> None:
        feature_dim = guide_cnn_feature_dim(sequence_length=26)
        model = GuideStrengthEncoder(
            feature_dim=feature_dim,
            architecture="cnn",
            sequence_length=26,
            cnn_channels=(8, 8),
            cnn_kernel_size=3,
        )
        x = torch.randn(3, feature_dim)

        out = model(x)

        self.assertEqual(out.shape, (3,))
        self.assertTrue(torch.all(out >= 0))


if __name__ == "__main__":
    unittest.main()
