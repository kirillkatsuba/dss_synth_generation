from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.imputation_comparison_report import (
    SCATTER_LIMITS,
    SCATTER_SCALES,
    DatasetSchema,
    deterministic_metrics,
    model_result_directory,
    normalized_energy_distance,
    pooled_target,
    register_dss_models,
    save_model_quality_summary,
    tv_complement,
)


class ImputationComparisonReportTest(unittest.TestCase):
    def test_hdd_scatter_uses_requested_fixed_linear_view(self) -> None:
        dataset = "dss_hdd_sequential_iops"
        self.assertEqual(SCATTER_SCALES[dataset], ("linear", "linear"))
        self.assertEqual(SCATTER_LIMITS[dataset]["iops"], (0.0, 14.0))
        self.assertEqual(SCATTER_LIMITS[dataset]["lat"], (0.0, 100.0))

    def test_model_result_directory_supports_both_layouts(self) -> None:
        root = Path("reports")
        self.assertEqual(
            model_result_directory("tabdiff", root, "dataset", "iops"),
            root / "dataset" / "loco" / "iops",
        )
        self.assertEqual(
            model_result_directory("dss_log", root, "dataset", "lat"),
            root / "loco_lat",
        )
        self.assertEqual(
            model_result_directory("dss_mixed_best", root, "dataset", "iops"),
            root / "loco_iops",
        )

    def test_register_dss_models_adds_arbitrary_evaluation_root(self) -> None:
        roots: dict[str, Path] = {}
        register_dss_models(
            [["dss_mixed_best", "DSS mixed (best)", "reports/test_best"]],
            roots,
        )
        self.assertEqual(
            roots["dss_mixed_best"], Path("reports/test_best").resolve()
        )

    def test_model_quality_summary_outputs_csv_markdown_and_png(self) -> None:
        keys = {
            "dataset_key": "dss_hdd_sequential_iops",
            "dataset": "HDD sequential",
            "model_key": "dss_log",
            "model": "DSS diffusion + log",
            "target": "iops",
            "status": "ok",
        }
        deterministic = pd.DataFrame([{**keys, "r2": 0.6, "nrmse": 0.4}])
        generative = pd.DataFrame(
            [
                {
                    **keys,
                    "trials": 10,
                    "target_shape_score": 0.9,
                    "target_context_pair_score": 0.8,
                    "target_c2st_quality": 0.7,
                    "target_alpha_precision": 0.85,
                    "target_beta_recall": 0.5,
                    "iops_lat_pair_score": 0.75,
                    "dcr_descriptive": 1.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            save_model_quality_summary(deterministic, generative, directory)
            self.assertTrue((directory / "model_quality_summary.csv").is_file())
            self.assertTrue((directory / "model_quality_summary.md").is_file())
            self.assertTrue((directory / "model_quality_summary.png").is_file())
            markdown = (directory / "model_quality_summary.md").read_text()
            self.assertIn("DSS diffusion + log", markdown)
            self.assertIn("0.600", markdown)

    def test_deterministic_numerical_metrics(self) -> None:
        real = pd.Series([1.0, 2.0, 3.0, 4.0])
        predicted = pd.Series([1.0, 2.0, 4.0, 3.0])
        metrics = deterministic_metrics(real, predicted, numerical=True)
        self.assertEqual(metrics["mse"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.5))
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["r2"], 0.6)

    def test_deterministic_categorical_metrics(self) -> None:
        real = pd.Series(["read", "read", "write", "write"])
        predicted = pd.Series(["read", "write", "write", "write"])
        metrics = deterministic_metrics(real, predicted, numerical=False)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertGreater(metrics["macro_f1"], 0.7)

    def test_pooled_target_selects_saved_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            pd.DataFrame({"iops": [1, 1, 1, 1]}).to_csv(
                directory / "trial_000.csv", index=False
            )
            pd.DataFrame({"iops": [2, 2, 2, 2]}).to_csv(
                directory / "trial_001.csv", index=False
            )
            pooled, trials = pooled_target(
                directory, "iops", rows=4, numerical=True, seed=7
            )
            self.assertEqual(trials, 2)
            self.assertTrue(set(pooled).issubset({1, 2}))

    def test_explicit_metadata_respects_schema(self) -> None:
        schema = DatasetSchema(
            columns=["iops", "iodepth", "io_type"],
            numerical={"iops", "iodepth"},
            categorical={"io_type"},
        )
        metadata = schema.metadata()
        self.assertEqual(metadata["columns"]["iodepth"]["sdtype"], "numerical")
        self.assertEqual(metadata["columns"]["io_type"]["sdtype"], "categorical")

    def test_tv_complement(self) -> None:
        real = pd.Series(["a", "a", "b", "b"])
        same = pd.Series(["a", "a", "b", "b"])
        shifted = pd.Series(["a", "a", "a", "a"])
        self.assertEqual(tv_complement(real, same), 1.0)
        self.assertEqual(tv_complement(real, shifted), 0.5)

    def test_energy_distance_is_positive_for_shifted_pairs(self) -> None:
        real = pd.DataFrame({"iops": [1.0, 2.0, 3.0], "lat": [3.0, 2.0, 1.0]})
        identical = real.copy()
        shifted = real + 10.0
        self.assertAlmostEqual(
            normalized_energy_distance(real, identical, size=0, seed=1), 0.0
        )
        self.assertGreater(
            normalized_energy_distance(real, shifted, size=0, seed=1), 0.0
        )


if __name__ == "__main__":
    unittest.main()
