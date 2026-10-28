from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff.data import PerturbationDataset


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_with_split.py"
_SPEC = importlib.util.spec_from_file_location("train_with_split_external_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ExternalActivityTrainingTest(unittest.TestCase):
    def _dataset(self) -> PerturbationDataset:
        return PerturbationDataset(
            np.ones((3, 2), dtype=np.float32),
            torch.tensor([0, 1, 2]),
            perturbation_labels=["control", "TP53_sg1", "MYC_sg2"],
            activity_labels=["NT_Control_1", "TP53_sg1", "MYC_sg2"],
        )

    def test_external_score_table_attaches_scalar_and_zero_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.csv"
            path.write_text(
                "guide_identity,sgrna_sequence,pam,virtual_scale\n"
                "TP53_sg1,ACGTACGTACGTACGTACGT,TGG,0.8\n"
                "MYC_sg2,GTACGTACGTACGTACGTAC,AGG,1.2\n",
                encoding="utf-8",
            )
            dataset = self._dataset()
            summary = _MODULE.attach_external_activity_features(
                dataset,
                backend="external_score",
                external_activity_table=path,
                external_activity_col="virtual_scale",
                activity_lookup=None,
            )

        self.assertEqual(summary["feature_dim"], 1)
        self.assertEqual(summary["matched_feature_cells"], 2)
        self.assertTrue(torch.allclose(dataset.guide_features[:, 0], torch.tensor([0.0, 0.8, 1.2])))

    def test_ruleset_backend_does_not_require_26mer_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "guides.csv"
            path.write_text(
                "guide_label,sequence,pam,target_genomic_sequence\n"
                "TP53_sg1,ACGTACGTACGTACGTACGT,TGG,ACGTACGTACGTACGTACGT\n"
                "MYC_sg2,GTACGTACGTACGTACGTAC,AGG,GTACGTACGTACGTACGTAC\n",
                encoding="utf-8",
            )
            dataset = self._dataset()
            summary = _MODULE.attach_external_activity_features(
                dataset,
                backend="crispri_ruleset",
                external_activity_table=path,
                external_activity_col="unused",
                activity_lookup=None,
            )

        self.assertEqual(summary["feature_dim"], 1)
        self.assertTrue(torch.all(dataset.guide_features[1:, 0] > 0))
        self.assertEqual(float(dataset.guide_features[0, 0]), 0.0)

    def test_external_score_rejects_missing_non_control_guide(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.tsv"
            path.write_text(
                "guide_identity\tsequence\tvirtual_scale\n"
                "TP53_sg1\tACGTACGTACGTACGTACGT\t0.8\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _MODULE.attach_external_activity_features(
                    self._dataset(),
                    backend="external_score",
                    external_activity_table=path,
                    external_activity_col="virtual_scale",
                    activity_lookup=None,
                )


if __name__ == "__main__":
    unittest.main()
