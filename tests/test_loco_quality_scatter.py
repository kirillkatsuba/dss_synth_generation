import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dss_loco_scatter_test_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dss_loco_scatter_test_cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plot_dss_imputer_evaluation import categorical_display_values
from scripts.plot_dss_imputer_loco_quality import pair_metric, plot_pair


class LocoQualityScatterTest(unittest.TestCase):
    def test_numeric_category_display_matches_int_and_float(self):
        values, support = categorical_display_values(pd.Series([128, 256, 128]),
                                                     pd.Series([128.0, 256.0, 128.0]), 12)
        np.testing.assert_equal(values["ground_truth"], values["model"])
        self.assertEqual(set(support), {"128", "256"})

    def test_pair_score_uses_same_semantic_numeric_categories(self):
        real = pd.DataFrame({"raid": ["a", "a", "b", "b"]*3, "n_disks": [1, 2, 1, 2]*3})
        generated = real.copy()
        generated["n_disks"] = generated.n_disks.astype(float)
        result = pair_metric(real, generated, "raid", "n_disks", set(), 10)
        self.assertEqual(result["error"], "")
        self.assertAlmostEqual(result["score"], 1)

    def test_scatter_mode_handles_all_pair_types_without_boxplots(self):
        real = pd.DataFrame({"iops": [20000, 40000, 50000, 60000], "lat": [1e6, 2e6, 3e6, 4e6],
                             "raid": ["a", "b", "a", "b"], "n_disks": [1, 2, 1, 2]})
        generated = real.copy()
        generated["n_disks"] = generated.n_disks.astype(float)
        with TemporaryDirectory() as directory, patch("scripts.plot_dss_imputer_loco_quality.sns.boxplot") as boxplot, \
                patch("scripts.plot_dss_imputer_loco_quality.sns.heatmap") as heatmap:
            for context, target in (("iops", "lat"), ("raid", "lat"), ("iops", "n_disks"), ("raid", "n_disks")):
                path = Path(directory) / f"{context}_{target}.png"
                plot_pair(real, generated, context, target, {"iops", "lat"}, path, 12, 10, {}, kind="scatter")
                self.assertGreater(path.stat().st_size, 1000)
            boxplot.assert_not_called()
            heatmap.assert_not_called()
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
