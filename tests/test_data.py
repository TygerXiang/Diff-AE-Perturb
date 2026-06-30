import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff.data import PerturbationDataset, encode_perturbation_labels


class PerturbationDataTest(unittest.TestCase):
    def test_encode_reserves_zero_for_controls(self) -> None:
        labels = ["neg_ctrl_non-targeting_0001", "TP53_sg1", "MYC_sg2", "TP53_sg3"]

        ids, mapping = encode_perturbation_labels(labels)

        self.assertEqual(ids.tolist()[0], 0)
        self.assertEqual(mapping["TP53"], 1)
        self.assertEqual(mapping["MYC"], 2)
        self.assertEqual(ids.dtype, torch.long)

    def test_raw_strategy_preserves_sgrna_labels(self) -> None:
        labels = ["non-targeting", "TP53_sg1", "TP53_sg2"]

        ids, mapping = encode_perturbation_labels(labels, target_strategy="raw")

        self.assertEqual(ids.tolist(), [0, 1, 2])
        self.assertEqual(mapping, {"TP53_sg1": 1, "TP53_sg2": 2})

    def test_encode_can_reuse_existing_mapping(self) -> None:
        labels = ["MYC_sg2", "neg_ctrl_non-targeting_0001", "TP53_sg1"]

        ids, mapping = encode_perturbation_labels(
            labels,
            existing_mapping={"TP53": 1, "MYC": 2},
        )

        self.assertEqual(ids.tolist(), [2, 0, 1])
        self.assertEqual(mapping, {"TP53": 1, "MYC": 2})

    def test_dataset_yields_model_batch_fields(self) -> None:
        expression = np.ones((3, 5), dtype=np.float32)
        ids = torch.tensor([0, 1, 2])
        dataset = PerturbationDataset(expression, ids)

        item = dataset[1]

        self.assertEqual(dataset.n_genes, 5)
        self.assertEqual(dataset.n_perturbations, 2)
        self.assertEqual(item["x0"].shape, (5,))
        self.assertEqual(item["perturbation_ids"].dtype, torch.long)
        self.assertNotIn("response_proxy", item)


if __name__ == "__main__":
    unittest.main()
