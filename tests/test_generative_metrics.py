from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from dss_imputer.generative_metrics import (
    _alpha_beta_embedding,
    aggregate_metric_trials,
    evaluate_generative_quality,
    generative_metric_seed,
    logistic_detection_quality,
    prepare_metric_tables,
    quality_report_scores,
)
from scripts.imputation_comparison_report import DatasetSchema, generative_metrics
from scripts.recompute_dss_imputer_metrics import (
    load_target_mask,
    main as recompute_main,
)


class GenerativeMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.real = pd.DataFrame(
            {
                "value": rng.normal(size=120),
                "n_disks": np.tile([1, 2, 4], 40),
                "raid": np.tile(["r0", "r1"], 60),
            }
        )
        self.generated = self.real.copy()
        self.generated["n_disks"] = np.roll(self.real.n_disks, 1)

    def evaluate(self, generated: pd.DataFrame, **kwargs):
        return evaluate_generative_quality(
            self.real,
            generated,
            self.real,
            sample_size=120,
            dcr_sample_size=30,
            seed=7,
            numerical_columns=["value"],
            categorical_columns=["n_disks", "raid"],
            **kwargs,
        )

    def test_numeric_categories_have_identical_scores_for_object_and_csv_dtypes(self):
        baseline, errors = self.evaluate(self.generated, target_columns=["n_disks"])
        as_object = self.generated.copy()
        as_object["n_disks"] = as_object.n_disks.astype(float).astype(object)
        actual, object_errors = self.evaluate(as_object, target_columns=["n_disks"])
        self.assertEqual(errors, [])
        self.assertEqual(object_errors, [])
        self.assertEqual(baseline, actual)
        self.assertEqual(actual["target_shape_valid_count"], 1)
        self.assertEqual(actual["target_pair_valid_count"], 2)
        self.assertLess(actual["target_column_pair_trends"], 1)

    def test_preparation_preserves_numeric_categories_unknowns_and_missingness(self):
        real = pd.DataFrame({"number": [1, 2, 3], "category": [1, 2, np.nan]})
        generated = pd.DataFrame(
            {"number": ["1", "2", "3"], "category": ["1.0", "9", None]}
        )
        original = generated.copy(deep=True)
        r, g, _, metadata = prepare_metric_tables(
            real,
            generated,
            numerical_columns=["number"],
            categorical_columns=["category"],
        )
        self.assertEqual(metadata["columns"]["category"]["sdtype"], "categorical")
        self.assertEqual(r.category.iloc[0], g.category.iloc[0])
        self.assertEqual(r.category.iloc[2], g.category.iloc[2])
        self.assertNotIn(g.category.iloc[1], r.category)
        self.assertEqual(g.number.dtype, np.dtype(float))
        pd.testing.assert_frame_equal(generated, original)

    def test_schema_and_row_mismatches_fail_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "exactly one type"):
            prepare_metric_tables(
                self.real,
                self.generated,
                numerical_columns=["value"],
                categorical_columns=["value", "raid", "n_disks"],
            )
        with self.assertRaisesRegex(ValueError, "matching columns and rows"):
            prepare_metric_tables(self.real, self.generated.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "infinity"):
            broken = self.generated.copy()
            broken.loc[0, "value"] = np.inf
            self.evaluate(broken)

    def test_internal_quality_errors_cannot_produce_perfect_partial_aggregates(self):
        report = Mock()
        shapes = pd.DataFrame(
            {"Column": ["value", "n_disks", "raid"], "Score": [1.0, 1.0, 1.0]}
        )
        pairs = pd.DataFrame(
            {
                "Column 1": ["value", "value", "n_disks"],
                "Column 2": ["n_disks", "raid", "raid"],
                "Score": [np.nan, 1.0, 0.8],
                "Error": ["TypeError: object dtype", None, None],
            }
        )
        report.get_details.side_effect = [shapes, pairs]
        scores, errors = quality_report_scores(report, ["n_disks"], self.real.columns)
        self.assertNotIn("column_pair_trends", scores)
        self.assertNotIn("overall_quality", scores)
        self.assertNotIn("target_column_pair_trends", scores)
        self.assertEqual(scores["target_pair_count"], 2)
        self.assertEqual(scores["target_pair_valid_count"], 1)
        self.assertTrue(any("object dtype" in error for error in errors))

    def test_missing_quality_pairs_are_not_silently_dropped(self):
        report = Mock()
        report.get_details.side_effect = [
            pd.DataFrame({"Column": self.real.columns, "Score": [1.0, 1.0, 1.0]}),
            pd.DataFrame(columns=["Column 1", "Column 2", "Score"]),
        ]
        scores, errors = quality_report_scores(report, ["n_disks"], self.real.columns)
        self.assertNotIn("overall_quality", scores)
        self.assertNotIn("target_overall_quality", scores)
        self.assertEqual(scores["target_pair_count"], 2)
        self.assertTrue(errors)

    def test_mcar_scores_only_hidden_rows_of_each_target(self):
        real = pd.DataFrame(
            {
                "context": np.tile(["a", "b", "c", "d"], 30),
                "target": np.tile([0, 1], 60),
            }
        )
        generated = real.copy()
        mask = np.zeros(real.shape, dtype=bool)
        mask[:60, 1] = True
        generated.loc[:59, "target"] = 1
        scores, errors = evaluate_generative_quality(
            real,
            generated,
            real,
            sample_size=120,
            dcr_sample_size=30,
            seed=7,
            numerical_columns=[],
            categorical_columns=list(real),
            target_mask=mask,
        )
        self.assertEqual(errors, [])
        self.assertEqual(scores["target_target_rows"], 60)
        self.assertAlmostEqual(scores["target_column_shapes"], 0.5)
        self.assertGreater(scores["column_shapes"], scores["target_column_shapes"])
        self.assertEqual(scores["target_target_pair_valid_count"], 1)

    def test_alpha_beta_numeric_category_encoding_respects_schema_and_unknown_support(
        self,
    ):
        real = pd.DataFrame({"category": [1, 2, 1, 2]})
        generated = pd.DataFrame({"category": [1, 2, 9, 9]}).astype(object)
        r, g = _alpha_beta_embedding(real, generated, [], ["category"])
        self.assertEqual(r.shape, (4, 3))
        self.assertTrue(np.all(r.sum(axis=1) == 1))
        self.assertTrue(np.all(g.sum(axis=1) == 1))
        self.assertFalse(np.any(np.all(r == g[2], axis=1)))

    def test_trial_aggregation_records_number_of_successful_trials(self):
        result = aggregate_metric_trials([{"score": 0.5}, {"score": np.nan}, {}])
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["score_trials"], 1)

    def test_metric_seed_depends_on_scenario_name(self):
        alone = generative_metric_seed(42, "loco_n_disks")
        together = {
            name: generative_metric_seed(42, name)
            for name in ("loco_iops", "loco_n_disks")
        }
        self.assertEqual(alone, together["loco_n_disks"])
        self.assertNotEqual(alone, together["loco_iops"])

    def test_c2st_preserves_generated_only_categories_without_crashing(self):
        real = pd.DataFrame({"category": [0, 100] * 60})
        generated = pd.DataFrame({"category": [0, 50, 100] * 40})
        r, g, _, metadata = prepare_metric_tables(
            real, generated, numerical_columns=[], categorical_columns=["category"]
        )
        score = logistic_detection_quality(r, g, metadata, seed=7)
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertEqual(score, logistic_detection_quality(r, g, metadata, seed=7))

    def test_comparison_report_uses_same_schema_for_all_metrics(self):
        schema = DatasetSchema(list(self.real), {"value"}, {"n_disks", "raid"})
        first = generative_metrics(
            self.real, self.generated, self.real, "n_disks", schema, 120, 30, 30, 7
        )
        generated = self.generated.copy()
        generated["n_disks"] = generated.n_disks.astype(float).astype(object)
        second = generative_metrics(
            self.real, generated, self.real, "n_disks", schema, 120, 30, 30, 7
        )
        self.assertEqual(first["metric_errors"], "")
        self.assertEqual(second["metric_errors"], "")
        self.assertEqual(first["target_context_pair_valid_count"], 2)
        for name in (
            "target_shape_score",
            "target_context_pair_score",
            "full_alpha_precision",
            "full_beta_recall",
            "full_c2st_quality",
            "dcr_descriptive",
        ):
            self.assertEqual(first[name], second[name])

    def test_recompute_uses_saved_trials_and_preserves_original_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            scenario = results / "loco_n_disks"
            scenario.mkdir(parents=True)
            self.real.to_csv(results / "evaluation_truth.csv", index=False)
            self.real.to_csv(root / "train.csv", index=False)
            self.generated.to_csv(scenario / "trial_000.csv", index=False)
            self.generated.astype({"n_disks": float}).to_csv(
                scenario / "trial_001.csv", index=False
            )
            (results / "run_info.json").write_text(
                json.dumps(
                    {
                        "model_columns": list(self.real),
                        "numerical_columns": ["value"],
                        "categorical_columns": ["n_disks", "raid"],
                        "rows": len(self.real),
                        "checkpoint": "missing.pt",
                        "seed": 7,
                    }
                )
            )
            original_summary = results / "generative_metrics_summary.csv"
            original_summary.write_text("original result\n")
            with patch.object(
                sys,
                "argv",
                [
                    "recompute_dss_imputer_metrics.py",
                    "--results-dir",
                    str(results),
                    "--train-reference-csv",
                    str(root / "train.csv"),
                    "--metric-sample-size",
                    "120",
                    "--dcr-sample-size",
                    "30",
                ],
            ):
                recompute_main()
            summary = pd.read_csv(
                results / "metrics_corrected" / "generative_metrics_summary.csv"
            )
            self.assertEqual(summary.trials.iloc[0], 2)
            self.assertEqual(summary.target_column_shapes_trials.iloc[0], 2)
            self.assertEqual(summary.target_n_disks_pair_valid_count.iloc[0], 2)
            self.assertEqual(original_summary.read_text(), "original result\n")

    def test_recompute_requires_original_mcar_mask(self):
        with TemporaryDirectory() as directory:
            scenario = Path(directory) / "mcar_0.15"
            scenario.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "original mask"):
                load_target_mask(scenario, self.real)
            mask = np.zeros(self.real.shape, dtype=bool)
            mask[:15, 1] = True
            pd.DataFrame(mask, columns=self.real.columns).to_csv(
                scenario / "target_mask.csv", index=False
            )
            np.testing.assert_array_equal(load_target_mask(scenario, self.real), mask)


if __name__ == "__main__":
    unittest.main()
