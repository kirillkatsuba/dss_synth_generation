import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dss_four_panel_tests/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dss_four_panel_tests/xdg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.export_loco_comparison import (
    MODEL_ORDER, NAMES, ORDER, best_scores, conditional_pair, distribution,
    export_minimal_report, marginal_scores, marginal_spec, metric_matrices, panel_axes, save_table,
    scatter_pair, table_html,
)


class FourPanelComparisonTest(unittest.TestCase):
    def test_exact_categorical_tv_includes_generated_only_levels(self):
        real = pd.Series(["a", "a", "b", "b"])
        generated = pd.Series(["a", "a", "c", "c"])
        scores = marginal_scores(real, generated, marginal_spec(real, "raid"))
        self.assertEqual(scores["tv_method"], "empirical_pmf")
        self.assertAlmostEqual(scores["tv"], .5)
        self.assertTrue(np.isnan(scores["ks"]))

    def test_numeric_tv_and_ks_do_not_round_predictions(self):
        real = pd.Series([1, 1, 2, 2])
        predicted = pd.Series([1.5, 1.5, 2.5, 2.5])
        scores = marginal_scores(real, predicted, marginal_spec(real, "block_size"))
        self.assertEqual(scores["tv"], 1)
        self.assertEqual(scores["ks"], .5)

    def test_continuous_tv_uses_gt_edges_and_counts_out_of_range_tail(self):
        real = pd.Series(np.linspace(.1, 1.1, 1000))
        spec = marginal_spec(real, "lat")
        self.assertEqual(spec["tv_method"], "gt_quantile_histogram")
        self.assertTrue(np.isneginf(spec["edges"][0]))
        self.assertTrue(np.isposinf(spec["edges"][-1]))
        predicted = real.copy()
        predicted.iloc[:100] = 1e9
        scores = marginal_scores(real, predicted, spec)
        self.assertAlmostEqual(scores["tv"], .1)
        self.assertAlmostEqual(scores["ks"], .1)
        identical = marginal_scores(real, real, spec)
        self.assertEqual(identical["tv"], 0)
        self.assertEqual(identical["ks"], 0)

    def test_constant_float_reference_and_nonfinite_predictions(self):
        real = pd.Series([3.14] * 10)
        spec = marginal_spec(real, "lat")
        self.assertEqual(spec["tv_method"], "empirical_pmf")
        self.assertEqual(marginal_scores(real, real + 1, spec)["tv"], 1)
        bad = marginal_scores(real, real.replace(3.14, np.inf), spec)
        self.assertTrue(np.isnan(bad["tv"]))
        self.assertIn("marginal_error", bad)

    def test_best_values_use_directions_groups_ties_and_warning_exclusion(self):
        frame = pd.DataFrame({"target": ["iops"] * 3 + ["lat"] * 3,
                              "tv": [.1, .2, .1, .9, .8, .7], "mse": [1, 2, 3, 3, 2, 1],
                              "r2": [-2, -.5, -1, .2, .3, .4], "c2st": [1, .9, .8, .5, .6, .7],
                              "c2st_convergence_warning": [True, False, False, False, False, False]})
        best = best_scores(frame)
        self.assertEqual(best.tv.tolist(), [True, False, True, False, False, True])
        self.assertEqual(best.r2.tolist(), [False, True, False, False, False, True])
        self.assertEqual(best.mse.tolist(), [True, False, False, False, False, True])
        self.assertEqual(best.c2st.tolist(), [False, True, False, False, False, True])

    def test_panel_order_has_one_gt_and_all_pair_types_render(self):
        fig, axes = panel_axes("Fixture", True)
        self.assertEqual(list(axes), list(ORDER))
        self.assertEqual([ax.get_title() for ax in axes.values()], [NAMES[key] for key in ORDER])
        self.assertEqual(sum(ax.get_title() == "Ground truth" for ax in fig.axes), 1)
        plt.close(fig)
        real = pd.DataFrame({"iops": [1000, 2000, 3000, 4000], "lat": [1e6, 2e6, 3e6, 4e6],
                             "raid": ["a", "b", "a", "b"], "block_size": [1, 2, 1, 2]})
        predictions = {key: real.copy() for key in MODEL_ORDER}
        predictions["tabsyn"].loc[0, "block_size"] = 3
        with TemporaryDirectory() as temp:
            root = Path(temp)
            distribution(real, predictions, "iops", root / "hist.png", "Fixture", True)
            distribution(real, predictions, "raid", root / "bars.png", "Fixture")
            for context, target in (("iops", "lat"), ("raid", "lat"), ("iops", "block_size"), ("raid", "block_size")):
                scatter_pair(real, predictions, target, context, "cache", root / f"scatter_{context}_{target}.png", "Fixture")
                if target == "block_size" or context == "raid":
                    conditional_pair(real, predictions, target, context, "cache", root / f"cond_{context}_{target}.png", "Fixture")
            self.assertTrue(all(path.stat().st_size > 1000 for path in root.glob("*.png")))
        self.assertEqual(plt.get_fignums(), [])

    def test_bold_html_png_and_per_metric_tables(self):
        frame = pd.DataFrame({"target": ["iops"] * 3, "model": MODEL_ORDER,
                              "tv": [.1, .3, .2], "ks": [.3, .2, .1], "r2": [.2, .5, .4]})
        rendered = table_html(frame)
        self.assertIn("<strong", rendered)
        self.assertIn("0.1000</strong>", rendered)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            save_table(frame, root / "metrics.png", "Fixture")
            # Real reports contain every field; missing fields must remain n/a.
            from scripts.export_loco_comparison import SCORES
            for field in SCORES:
                if field not in frame: frame[field] = np.nan
            metric_matrices(frame, root / "tables", "Fixture")
            self.assertTrue((root / "tables/tv.png").is_file())
            self.assertTrue((root / "tables/index.html").is_file())
        self.assertEqual(plt.get_fignums(), [])

    def test_minimal_export_contains_only_scatter_distributions_and_generative_tables(self):
        from scripts.export_loco_comparison import SCORES
        with TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "minimal"
            source.mkdir()
            records = []
            for split in ("train", "test"):
                for target, context in (("iops", "lat"), ("lat", "iops")):
                    directory = source / "cache" / split / "main" / f"loco_{target}"
                    directory.mkdir(parents=True)
                    for name in ("distribution.png", f"pair_{context}.png", "conditional_extra.png", "distribution_log.png"):
                        (directory / name).write_bytes(b"plot fixture")
                    for index, model in enumerate(MODEL_ORDER):
                        records.append({"dataset": "cache", "split": split, "variant": "main",
                                        "target": target, "model": model, "rows": 100, "metric_rows": 100,
                                        "c2st_convergence_warning": False,
                                        **{field: .1 * (index + 1) for field in SCORES}})
            pd.DataFrame(records).to_csv(source / "metrics_all.csv", index=False)
            export_minimal_report(source, output, ["cache"], ["train", "test"])
            for target in ("iops", "lat"):
                directory = output / "cache" / f"loco_{target}"
                self.assertTrue(directory.is_dir())
                self.assertEqual(len(list(directory.glob("*.png"))), 6)
                self.assertFalse(any("conditional" in path.name or "log" in path.name for path in directory.iterdir()))
                table = pd.read_csv(directory / "train_metrics.csv")
                self.assertNotIn("mse", table)
                self.assertNotIn("r2", table)
                self.assertIn("tv", table)
                self.assertIn("ks", table)
                page = (directory / "index.html").read_text()
                self.assertIn("<u>", page)
                self.assertNotIn("<strong", page)
            self.assertFalse((output / "cache" / "train").exists())
            with self.assertRaises(ValueError):
                export_minimal_report(source, source, ["cache"], ["test"])

    def test_underlined_table_excludes_deterministic_metrics(self):
        from scripts.export_loco_comparison import SCORES
        frame = pd.DataFrame({"target": ["iops"] * 3, "model": MODEL_ORDER,
                              **{field: [.1, .2, .3] for field in SCORES}})
        fields = [field for field in SCORES if field not in {"mse", "r2"}]
        rendered = table_html(frame, fields=fields, emphasis="underline")
        self.assertIn("<u>0.1000</u>", rendered)
        self.assertNotIn("MSE", rendered)
        self.assertNotIn("R2", rendered)
        self.assertNotIn("<strong", rendered)
        with TemporaryDirectory() as temp:
            output = Path(temp) / "underlined.png"
            save_table(frame, output, "Fixture", fields=fields, emphasis="underline")
            self.assertGreater(output.stat().st_size, 1000)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
