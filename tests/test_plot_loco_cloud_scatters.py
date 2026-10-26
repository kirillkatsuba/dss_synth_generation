import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.plot_loco_cloud_scatters import display_axis, load_aligned_cloud, plot_target_pairs


class CloudScatterTest(unittest.TestCase):
    def test_numeric_axis_keeps_outliers_and_uses_linear_scale(self):
        values, limits, ticks = display_axis([pd.Series([1000, 2000]), pd.Series([0, 100000])], "iops")
        np.testing.assert_equal(values[1], [0, 100])
        self.assertLess(limits[0], 0)
        self.assertGreater(limits[1], 100)
        self.assertIsNone(ticks)

    def test_categories_share_positions_and_ticks(self):
        values, _, ticks = display_axis([pd.Series(["8+2", "8+2"]), pd.Series(["4+1", "8+2"])], "raid")
        self.assertEqual(ticks, ["4+1", "8+2"])
        np.testing.assert_equal(values[0], [1, 1])
        np.testing.assert_equal(values[1], [0, 1])

    def test_nonfinite_values_fail_instead_of_dropping_rows(self):
        with self.assertRaises(ValueError):
            display_axis([pd.Series([1, np.inf])], "lat")

    def test_aligns_by_source_row_and_checks_context(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            truth = pd.DataFrame({"source_row": [5, 8], "cloud_id": [1, 1],
                                  "cloud_name": ["Cloud 1", "Cloud 1"], "id": ["a", "a"],
                                  "iops": [1000, 2000], "lat": [1e6, 2e6], "raid": ["8+2", "8+2"]})
            prediction = truth.copy()
            prediction["ground_truth_iops"] = prediction.iops
            prediction["iops"] = [900, 1900]
            truth.to_csv(root / "truth.csv", index=False)
            prediction.iloc[::-1].to_csv(root / "prediction.csv", index=False)
            real, generated = load_aligned_cloud(root / "truth.csv", root / "prediction.csv", "iops")
            self.assertEqual(generated.source_row.tolist(), [5, 8])
            plot_target_pairs(real, generated, "iops", ["iops", "lat", "raid"], root / "plots/cloud.png", 50)
            self.assertGreater((root / "plots/cloud.png").stat().st_size, 1000)
            prediction.loc[0, "lat"] = 3e6
            prediction.to_csv(root / "prediction.csv", index=False)
            with self.assertRaises(ValueError):
                load_aligned_cloud(root / "truth.csv", root / "prediction.csv", "iops")


if __name__ == "__main__":
    unittest.main()
