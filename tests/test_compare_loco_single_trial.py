import unittest
import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dss_single_trial_test/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dss_single_trial_test/xdg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.compare_loco_single_trial import (
    MODELS, Source, align_sources, category_values, load_prediction, metric_record,
    save_conditional_pair, save_distribution, save_pair,
)


class SingleTrialComparisonTest(unittest.TestCase):
    def test_alignment_handles_float32_subset_reordering_and_duplicates(self):
        truth = pd.DataFrame({"iops": [10, 20, 10], "lat": [18255364.57, 4.13, 18255364.57]})
        external = pd.concat([truth.iloc[[1, 0, 2]], pd.DataFrame({"iops": [30], "lat": [8]})], ignore_index=True).astype(np.float32)
        sources = [Source("hybrid", Path("hybrid"), {}, truth),
                   Source("tabsyn", Path("tabsyn"), {}, external),
                   Source("tabdiff", Path("tabdiff"), {}, external.iloc[[3, 0, 1, 2]].reset_index(drop=True))]
        shared, audit = align_sources(sources, list(truth))
        pd.testing.assert_frame_equal(shared, truth)
        np.testing.assert_equal(sources[1].indexer, [1, 0, 2])
        self.assertEqual(audit[1]["source_rows_not_compared"], 1)

    def test_context_checked_against_own_gt_then_precision_harmonized(self):
        truth = pd.DataFrame({"iops": [10, 20], "lat": [18255364.57, 4.13]})
        external = truth.astype(np.float32)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "loco_iops").mkdir()
            prediction = external.copy()
            prediction["iops"] = [12.25, 21.75]
            path = root / "loco_iops/trial_000.csv"
            prediction.to_csv(path, index=False)
            source = Source("tabsyn", root, {}, pd.read_csv(path).assign(iops=external.iops), np.arange(2))
            output, checks = load_prediction(source, list(truth), "iops", 0, truth)
            np.testing.assert_array_equal(output.lat, truth.lat)
            np.testing.assert_array_equal(output.iops, prediction.iops)
            self.assertEqual(sum(row["mismatches"] for row in checks), 0)
            prediction.loc[0, "lat"] = 0
            prediction.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Known context changed"):
                load_prediction(source, list(truth), "iops", 0, truth)

    @patch("scripts.compare_loco_single_trial.evaluate_generative_quality", return_value=({}, []))
    def test_numeric_category_mse_is_not_rounded_and_text_has_no_r2(self, evaluate):
        truth = pd.DataFrame({"block_size": [1, 2, 3], "raid": ["a", "b", "a"]})
        prediction = truth.assign(block_size=[1.5, 2, 2.5])
        scores, _ = metric_record(truth, prediction, "block_size", 100, 42)
        self.assertAlmostEqual(scores["mse"], 1 / 6)
        self.assertAlmostEqual(scores["r2"], .75)
        self.assertAlmostEqual(scores["outside_gt_support_fraction"], 2 / 3)
        self.assertEqual(evaluate.call_args.kwargs["categorical_columns"], ["block_size", "raid"])
        scores, _ = metric_record(truth, prediction, "raid", 100, 42)
        self.assertTrue(np.isnan(scores["mse"]))
        self.assertTrue(np.isnan(scores["r2"]))

    def test_plots_cover_all_pair_types(self):
        truth = pd.DataFrame({"iops": [1000, 2000, 3000, 4000], "lat": [1e6, 2e6, 3e6, 4e6],
                              "raid": ["a", "b", "a", "b"], "block_size": [1, 2, 1, 2]})
        predictions = {model: truth.copy() for model in MODELS}
        predictions["tabsyn"].loc[0, "block_size"] = 3
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for context, target in (("iops", "lat"), ("raid", "lat"), ("iops", "block_size"), ("raid", "block_size")):
                output = root / f"{context}_{target}.png"
                save_conditional_pair(truth, predictions, target, context, "cache", output, "Fixture")
                self.assertGreater(output.stat().st_size, 1000)
            save_distribution(truth, predictions, "iops", root / "hist.png", "Fixture", True)
            save_pair(truth, predictions, "block_size", "raid", "cache", root / "scatter.png", "Fixture")
        self.assertEqual(plt.get_fignums(), [])

    def test_display_grouping_keeps_unknown_separate_from_rare_gt(self):
        real = pd.Series([1, 1, 2, 3])
        arrays, labels = category_values(real, [pd.Series([1.0, 2.0, 9.0, 3.0])], 1)
        self.assertEqual(labels, ["1", "Other GT", "Outside GT"])
        np.testing.assert_array_equal(arrays[1], [0, 1, 2, 1])

    @patch("scripts.compare_loco_single_trial.evaluate_generative_quality")
    def test_estimator_warning_is_preserved_in_metric_audit(self, evaluate):
        def warn(*args, **kwargs):
            warnings.warn("Estimator reached its iteration limit", RuntimeWarning)
            return {}, []
        evaluate.side_effect = warn
        truth = pd.DataFrame({"iops": [1, 2, 3]})
        _, errors = metric_record(truth, truth, "iops", 100, 42)
        self.assertEqual(errors, ["warning/RuntimeWarning: Estimator reached its iteration limit"])


if __name__ == "__main__":
    unittest.main()
