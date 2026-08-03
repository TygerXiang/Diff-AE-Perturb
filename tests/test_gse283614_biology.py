import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perturbdiff.gse283614_biology import (
    benjamini_hochberg,
    build_rank_stratified_guide_folds,
    fit_response_models,
    hill_curve,
    validate_dosage_coverage,
    validate_fold_partition,
)


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GSE283614BiologyTest(unittest.TestCase):
    def test_synthetic_linear_quadratic_and_hill_are_recovered(self) -> None:
        x = np.linspace(0.1, 1.1, 24)
        linear = fit_response_models(x, 0.1 + 0.8 * x, models=("linear", "hill"))
        quadratic = fit_response_models(x, 1.0 - 3.0 * np.square(x - 0.55))
        hill = fit_response_models(x, hill_curve(x, 0.5, 4.0), models=("linear", "hill"))

        self.assertEqual(next(row["model"] for row in linear if row["selected_aic"]), "linear")
        quadratic_winner = next(row for row in quadratic if row["selected_aic"])
        self.assertEqual(quadratic_winner["model"], "quadratic")
        self.assertTrue(quadratic_winner["inverted_u"])
        self.assertAlmostEqual(float(quadratic_winner["vertex"]), 0.55, places=2)
        hill_winner = next(row for row in hill if row["selected_aic"])
        self.assertEqual(hill_winner["model"], "hill")
        self.assertTrue(hill_winner["threshold_in_observed_range"])
        self.assertGreater(float(hill_winner["hill_coefficient"]), 1.0)

    def test_rank_stratified_folds_cover_each_target_without_leakage(self) -> None:
        rows = []
        for target in ("NANOG", "OCT4"):
            for idx in range(10):
                rows.append(
                    {
                        "dose_target": target,
                        "guide_identity": f"{target}_{idx}",
                        "knockdown_fraction": idx / 10,
                        "is_control": False,
                    }
                )
        assignments = build_rank_stratified_guide_folds(pd.DataFrame(rows), n_folds=5)
        self.assertEqual(set(assignments.values()), set(range(5)))
        for fold in range(5):
            test = {guide for guide, value in assignments.items() if value == fold}
            val = {guide for guide, value in assignments.items() if value == (fold + 1) % 5}
            train = set(assignments) - test - val
            validate_fold_partition(train, val, test)
            self.assertEqual({guide.split("_", 1)[0] for guide in test}, {"NANOG", "OCT4"})

    def test_fold_validation_detects_leakage(self) -> None:
        with self.assertRaises(ValueError):
            validate_fold_partition({"g1", "g2"}, {"g2"}, {"g3"})

    def test_coverage_requires_all_guides(self) -> None:
        dosage = pd.DataFrame(
            {
                "guide_identity": ["g1", "g2", "c1", "c1"],
                "is_control": [False, False, True, True],
            }
        )
        summary = validate_dosage_coverage(
            dosage,
            ["g1", "g2", "c1"],
            expected_total_guides=3,
            expected_targeting_guides=2,
        )
        self.assertEqual(summary["n_targeting_guides"], 2)
        with self.assertRaises(ValueError):
            validate_dosage_coverage(
                dosage,
                ["g1", "g2", "c1", "missing"],
                expected_total_guides=4,
                expected_targeting_guides=2,
            )

    def test_bh_adjustment_is_monotone_in_pvalue_order(self) -> None:
        pvalues = np.asarray([0.04, 0.001, 0.02, np.nan])
        adjusted = benjamini_hochberg(pvalues)
        order = np.argsort(pvalues[:3])
        self.assertTrue(np.all(np.diff(adjusted[:3][order]) >= 0))
        self.assertTrue(np.isnan(adjusted[3]))

    def test_direct_activity_analysis_passes_feature_tensor(self) -> None:
        module = load_script("analyze_gse283614_dosage.py")

        class FakeEmbedding:
            weight = torch.ones(2, 3)

        class FakePerturbation:
            embedding = FakeEmbedding()

        class FakeModel:
            guide_doser = object()
            perturbation = FakePerturbation()

            def __init__(self):
                self.features = None

            def sample_counterfactual(self, controls, target_ids, *, guide_features, steps):
                self.features = guide_features.detach().cpu()
                return controls

        model = FakeModel()
        controls = np.ones((4, 3), dtype=np.float32)
        mean, _ = module.predict_mean(
            model,
            {"NANOG": 1},
            controls,
            guide_label="NANOG_1-1",
            target_label="NANOG",
            activity_value=0.65,
            steps=2,
            device=torch.device("cpu"),
        )
        self.assertTrue(np.allclose(mean, 1.0))
        self.assertTrue(torch.allclose(model.features, torch.full((4, 1), 0.65)))
        with self.assertRaises(ValueError):
            module.predict_mean(
                model,
                {"NANOG": 1},
                controls,
                guide_label="NANOG_1-1",
                target_label="NANOG",
                activity_value=None,
                steps=2,
                device=torch.device("cpu"),
            )

    def test_direct_training_uses_activity_labels_and_rejects_missing_guides(self) -> None:
        module = load_script("train_with_split.py")
        from perturbdiff.data import PerturbationDataset

        dataset = PerturbationDataset(
            np.ones((3, 2), dtype=np.float32),
            torch.tensor([0, 1, 2]),
            perturbation_labels=["control", "NANOG", "OCT4"],
            activity_labels=["NT_Control_1", "NANOG_1-1", "OCT4_1-38"],
        )
        with self.assertRaises(ValueError):
            module.attach_direct_activity_features(dataset, activity_lookup={"NANOG_1-1": 0.5})

        summary = module.attach_direct_activity_features(
            dataset,
            activity_lookup={"NANOG_1-1": 0.5, "OCT4_1-38": 0.8},
        )
        self.assertEqual(summary["missing_activity_cells"], 0)
        self.assertTrue(torch.allclose(dataset.guide_features[:, 0], torch.tensor([0.0, 0.5, 0.8])))

    def test_control_partition_is_disjoint_and_complete(self) -> None:
        module = load_script("create_gse283614_dose_cv.py")
        partition = module.control_partition(np.arange(101), seed=7)
        train = set(partition["train"].tolist())
        val = set(partition["val"].tolist())
        test = set(partition["test"].tolist())
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(train | val | test, set(range(101)))

    def test_oof_buffering_and_strong_suppression_checks(self) -> None:
        module = load_script("evaluate_gse283614_dose_cv.py")
        essential_genes = [f"ESS_{idx}" for idx in range(8)]
        lineage_genes = [f"LIN_{idx}" for idx in range(8)]
        genes = essential_genes + lineage_genes
        vectors = {}
        for method in ("observed", "target_only", "dose"):
            for state in ("ESC", "DE"):
                for target in ("NANOG", "OCT4"):
                    for idx, (essential, lineage) in enumerate(((1.0, 1.0), (1.05, 2.0), (0.95, 3.0))):
                        vectors[(method, state, target, f"g{idx}")] = [
                            np.asarray([essential] * len(essential_genes) + [lineage] * len(lineage_genes))
                        ]
        buffering = module.essentiality_buffering(
            vectors,
            genes,
            {
                "essentiality": essential_genes,
                "pluripotency": lineage_genes,
                "differentiation": lineage_genes,
            },
        )
        self.assertTrue(buffering["essential_buffered"].all())
        self.assertTrue(buffering["essential_buffered_significant"].all())

        frame = pd.DataFrame(
            {
                "state": ["DE"] * 8,
                "dose_target": ["NANOG"] * 4 + ["OCT4"] * 4,
                "knockdown_fraction": [0.1, 0.4, 0.7, 0.95] * 2,
                "dose_de_gain": [0.1, 0.8, 0.7, 0.2] * 2,
                "dose_ne_gain": [0.0, 0.1, 0.2, 0.9] * 2,
            }
        )
        checks = module.strong_suppression_checks(frame)
        self.assertTrue(all(row["de_declines_and_ne_rises"] for row in checks))


if __name__ == "__main__":
    unittest.main()
