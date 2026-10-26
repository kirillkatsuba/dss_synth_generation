from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.plot_dss_imputer_evaluation import (
    density_transform,
    discrete_bin_edges,
    is_discrete_integer,
    plot_distribution,
    plot_categorical_confusion,
    plot_row_comparison,
    resolve_data_csv,
    scaled_values,
    target_divisor,
)


class PlotDssImputerEvaluationTest(unittest.TestCase):
    def test_latency_uses_expected_display_scale(self) -> None:
        values = scaled_values(pd.Series([1e6, 2e6]), "lat")
        np.testing.assert_allclose(values, [1.0, 2.0])
        self.assertEqual(target_divisor("lat"), 1e6)

    def test_latency_density_transform_keeps_zero(self) -> None:
        transformed, label = density_transform(np.array([0.0, 9.0]), "lat")
        np.testing.assert_allclose(transformed, [0.0, 1.0])
        self.assertIn("log10", label)

    def test_latency_density_transform_supports_linear_scale(self) -> None:
        transformed, label = density_transform(
            np.array([0.0, 9.0]), "lat", density_scale="linear"
        )
        np.testing.assert_allclose(transformed, [0.0, 9.0])
        self.assertEqual(label, "Latency ($\\mu$s)")

    def test_low_cardinality_integers_use_histograms(self) -> None:
        self.assertTrue(
            is_discrete_integer(
                [np.array([1.0, 2.0, 4.0]), np.array([1.0, 4.0])],
                max_discrete_values=50,
            )
        )
        self.assertFalse(
            is_discrete_integer(
                [np.linspace(0.0, 100.0, 200)], max_discrete_values=50
            )
        )
        np.testing.assert_allclose(
            discrete_bin_edges(np.array([1.0, 2.0, 4.0])),
            [0.5, 1.5, 3.0, 5.0],
        )

    def test_distribution_plots_support_integer_and_categorical_targets(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plot_distribution(
                pd.Series([1, 2, 4, 1, 2]),
                pd.Series([1, 2, 2, 1, 4]),
                "iodepth",
                "numerical",
                root / "integer",
                "model",
                "train",
                sample_size=100,
                max_discrete_values=50,
                seed=7,
            )
            plot_distribution(
                pd.Series([0.5, 1.0, 2.0, 3.0]),
                pd.Series([0.4, 1.2, 2.2, 2.8]),
                "iops",
                "numerical",
                root / "continuous_histogram",
                "model",
                "train",
                sample_size=100,
                max_discrete_values=2,
                seed=7,
                density_scale="linear",
                display_limits=(0.0, 4.0),
                histogram_columns={"iops"},
                histogram_bins=8,
            )
            plot_distribution(
                pd.Series(["read", "write", "read"]),
                pd.Series(["read", "read", "write"]),
                "io_type",
                "categorical",
                root / "categorical",
                "model",
                "train",
                sample_size=100,
                max_discrete_values=50,
                seed=7,
            )
            self.assertTrue((root / "integer" / "comparison.png").is_file())
            self.assertTrue((root / "categorical" / "comparison.png").is_file())
            self.assertTrue(
                (root / "continuous_histogram" / "comparison.png").is_file()
            )

    def test_pairwise_plots_cover_numerical_and_categorical_columns(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plot_row_comparison(
                pd.Series([1.0, 2.0, 3.0]),
                pd.Series([1.2, 2.2, 2.8]),
                "n_jobs",
                "numerical",
                root / "numerical_rows.png",
                sample_size=3,
                seed=7,
            )
            truth = pd.Series(["read", "write", "read", "read"])
            prediction = pd.Series(["read", "read", "write", "read"])
            plot_row_comparison(
                truth, prediction, "io_type", "categorical",
                root / "categorical_rows.png", sample_size=4, seed=7,
            )
            plot_categorical_confusion(
                truth, prediction, "io_type", root / "confusion.png", 20
            )
            for name in ["numerical_rows.png", "categorical_rows.png", "confusion.png"]:
                self.assertTrue((root / name).is_file(), name)

    def test_saved_evaluation_rows_override_original_csv(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / "evaluation_truth.csv"
            saved.write_text("column\nvalue\n")
            args = Namespace(results_dir=root, data_csv=root / "original.csv", test_csv=None)
            self.assertEqual(resolve_data_csv(args, {}), saved)


if __name__ == "__main__":
    unittest.main()
