import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.export_loco_clouds import export_groups, validate_alignment


class LocoCloudExportTest(unittest.TestCase):
    def setUp(self):
        self.truth = pd.DataFrame({"iops": [10, 20, 30], "lat": [4.0, 5.0, 6.0],
                                   "raid": ["8+2", "4+1", "8+2"], "id": ["a", "b", "a"]})
        self.prediction = self.truth.copy()
        self.prediction["iops"] = [11, 21, 29]

    def test_accepts_only_target_changes(self):
        validate_alignment(self.truth, self.prediction, "iops")
        self.prediction.loc[1, "lat"] = 9
        with self.assertRaises(ValueError):
            validate_alignment(self.truth, self.prediction, "iops")

    def test_rejects_reordered_or_truncated_rows(self):
        for frame in (self.prediction.iloc[::-1].reset_index(drop=True), self.prediction.iloc[:2]):
            with self.assertRaises(ValueError):
                validate_alignment(self.truth, frame, "iops")

    def test_export_keeps_fixed_membership_and_truth_target(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            export_groups(self.prediction, np.array([1, 0, 1]),
                          {1: {"name": "Cloud 1"}}, root, "iops", self.truth)
            actual = pd.read_csv(root / "cloud_01.csv")
            self.assertEqual(actual.source_row.tolist(), [0, 2])
            self.assertEqual(actual.iops.tolist(), [11, 29])
            self.assertEqual(actual.ground_truth_iops.tolist(), [10, 30])
            pd.testing.assert_frame_equal(actual, pd.read_csv(root / "all_clouds.csv"))


if __name__ == "__main__":
    unittest.main()
