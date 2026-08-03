import unittest
from unittest.mock import patch

import pandas as pd
from scipy import sparse

from perturbdiff.gse283614 import (
    GuideCallConfig,
    build_log2fc_relative_activity_table,
    build_table_s1_activity_table,
    call_guides_from_matrix,
    guide_family,
    guide_target_gene,
    guide_type,
    normalize_guide_label,
    table_s1_name_to_guide_identity,
)


class GSE283614GuideTest(unittest.TestCase):
    def test_normalize_unifies_state_specific_guide_names(self) -> None:
        self.assertEqual(normalize_guide_label("OCT4+1-38"), "OCT4_1-38")
        self.assertEqual(normalize_guide_label("OCT4_1-38"), "OCT4_1-38")
        self.assertEqual(normalize_guide_label("OCT4-pro-1"), "OCT4-pro1")
        self.assertEqual(normalize_guide_label("OCT4-pro1"), "OCT4-pro1")
        self.assertEqual(normalize_guide_label("NT_Control-1"), "NT_Control_1")

    def test_guide_family_and_target_gene(self) -> None:
        self.assertEqual(guide_family("OCT4+1-38"), "OCT4")
        self.assertEqual(guide_target_gene("OCT4+1-38"), "POU5F1")
        self.assertEqual(guide_family("NANOG_1-1"), "NANOG")
        self.assertEqual(guide_target_gene("NANOG_1-1"), "NANOG")
        self.assertEqual(guide_family("SafeHarborchr1_1007"), "control_safeharbor")
        self.assertEqual(guide_type("NANOG-pro1"), "promoter")

    def test_table_s1_media_names_map_to_guide_capture_labels(self) -> None:
        oct4_row = pd.Series(
            {
                "table_s1_type": "OCT4",
                "region": "OCT4-DE",
                "table_s1_name": "OCT4-DE1",
            }
        )
        nanog_row = pd.Series(
            {
                "table_s1_type": "NANOG",
                "region": "NANOG e1",
                "table_s1_name": "NANOG-e1-2",
            }
        )

        self.assertEqual(table_s1_name_to_guide_identity(oct4_row), ("OCT4_1-38", "region_order"))
        self.assertEqual(table_s1_name_to_guide_identity(nanog_row), ("NANOG_1-1", "region_order"))

    def test_call_guides_from_capture_matrix(self) -> None:
        guide_names = ["OCT4+1-38", "NT_Control-1", "NANOG+1-1"]
        matrix = sparse.csr_matrix(
            [
                [10, 0, 0],
                [0, 4, 1],
                [1, 1, 0],
                [0, 0, 0],
            ]
        )

        calls = call_guides_from_matrix(
            matrix,
            guide_names,
            config=GuideCallConfig(
                min_total_umis=2,
                min_top_umis=2,
                min_top_fraction=0.5,
                min_top_ratio=1.5,
            ),
        )

        self.assertEqual(calls.loc[0, "guide_identity"], "OCT4_1-38")
        self.assertEqual(calls.loc[0, "perturbation_label"], "OCT4_1-38")
        self.assertEqual(calls.loc[0, "guide_assignment_status"], "assigned_guide")
        self.assertEqual(calls.loc[1, "perturbation_label"], "control")
        self.assertEqual(calls.loc[1, "guide_assignment_status"], "assigned_control")
        self.assertEqual(calls.loc[2, "guide_assignment_status"], "ambiguous")
        self.assertEqual(calls.loc[3, "guide_assignment_status"], "no_guide")

    def test_table_s1_z_scores_convert_to_nonnegative_relative_activity(self) -> None:
        rows = [
            {
                "table_s1_type": "OCT4",
                "region": "OCT4-DE",
                "group": "g",
                "table_s1_name": "DE-1",
                "sequence": "ACGT",
                "starting_position_hg38": "1",
                "screen_z_score_ESC": -4.0,
                "screen_z_score_DE": 2.0,
            },
            {
                "table_s1_type": "OCT4",
                "region": "OCT4-DE",
                "group": "g",
                "table_s1_name": "DE-2",
                "sequence": "ACGA",
                "starting_position_hg38": "2",
                "screen_z_score_ESC": -2.0,
                "screen_z_score_DE": -1.0,
            },
            {
                "table_s1_type": "OCT4",
                "region": "OCT4-PE",
                "group": "g",
                "table_s1_name": "PE-1",
                "sequence": "ACGG",
                "starting_position_hg38": "3",
                "screen_z_score_ESC": -8.0,
                "screen_z_score_DE": -2.0,
            },
        ]
        with patch("perturbdiff.gse283614.read_table_s1_grna_library", return_value=pd.DataFrame(rows)):
            activity = build_table_s1_activity_table("unused.xlsx")

        self.assertEqual(activity.loc[0, "guide_identity"], "OCT4_1-38")
        self.assertEqual(activity.loc[1, "guide_identity"], "OCT4_1-78")
        self.assertEqual(activity.loc[0, "screen_z_score_ESC"], -4.0)
        self.assertEqual(activity.loc[0, "screen_depletion_effect_ESC"], 4.0)
        self.assertEqual(activity.loc[0, "screen_strength_ESC"], 4.0)
        self.assertEqual(activity.loc[0, "screen_strength_DE"], 0.0)
        self.assertEqual(activity.loc[0, "activity_reference_effect_ESC"], 4.0)
        self.assertEqual(activity.loc[0, "relative_activity_ESC"], 1.0)
        self.assertEqual(activity.loc[1, "relative_activity_ESC"], 0.5)
        self.assertEqual(activity.loc[0, "relative_activity_signed_DE"], -2.0)
        self.assertEqual(activity.loc[1, "relative_activity_DE"], 1.0)

    def test_reference_log2fc_converts_to_relative_activity_by_sequence(self) -> None:
        reference_rows = pd.DataFrame(
            [
                {
                    "reference_region": "OCT4e+1 (OCT4 DE)",
                    "reference_grna_name": "OCT4e+1-38",
                    "sequence": "AAAA",
                    "screen_log2fc_day9_day0": -4.0,
                },
                {
                    "reference_region": "OCT4e+1 (OCT4 DE)",
                    "reference_grna_name": "OCT4e+1-78",
                    "sequence": "CCCC",
                    "screen_log2fc_day9_day0": -2.0,
                },
                {
                    "reference_region": "OCT4e+1 (OCT4 DE)",
                    "reference_grna_name": "OCT4e+1-17",
                    "sequence": "GGGG",
                    "screen_log2fc_day9_day0": 1.0,
                },
            ]
        )
        guide_rows = pd.DataFrame(
            [
                {
                    "table_s1_type": "OCT4",
                    "region": "OCT4-DE",
                    "group": "g",
                    "table_s1_name": "OCT4-DE1",
                    "sequence": "AAAA",
                    "starting_position_hg38": "1",
                    "screen_z_score_ESC": -1.0,
                    "screen_z_score_DE": -1.0,
                },
                {
                    "table_s1_type": "OCT4",
                    "region": "OCT4-DE",
                    "group": "g",
                    "table_s1_name": "OCT4-DE2",
                    "sequence": "CCCC",
                    "starting_position_hg38": "2",
                    "screen_z_score_ESC": -1.0,
                    "screen_z_score_DE": -1.0,
                },
                {
                    "table_s1_type": "OCT4",
                    "region": "OCT4-DE",
                    "group": "g",
                    "table_s1_name": "OCT4-DE5",
                    "sequence": "GGGG",
                    "starting_position_hg38": "3",
                    "screen_z_score_ESC": -1.0,
                    "screen_z_score_DE": -1.0,
                },
            ]
        )

        with patch("perturbdiff.gse283614.read_log2fc_reference_grna_table", return_value=reference_rows), patch(
            "perturbdiff.gse283614.read_table_s1_grna_library",
            return_value=guide_rows,
        ):
            activity = build_log2fc_relative_activity_table(
                "reference.xlsx",
                guide_library_path="media-1.xlsx",
            )

        by_guide = activity.set_index("guide_identity")
        self.assertEqual(by_guide.loc["OCT4_1-38", "relative_activity"], 1.0)
        self.assertEqual(by_guide.loc["OCT4_1-78", "relative_activity"], 0.5)
        self.assertEqual(by_guide.loc["OCT4_1-17", "relative_activity"], 0.0)
        self.assertEqual(by_guide.loc["OCT4_1-78", "activity_reference_grna_name"], "OCT4e+1-38")


if __name__ == "__main__":
    unittest.main()
