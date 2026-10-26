import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.clean_dataset_outliers import clean_clouds, main
from scripts.training_data import dataset_paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fixture() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    rows = []
    for group, io_type, center in (("first", "read", [8, 13]), ("second", "write", [12, 9])):
        log_values = rng.multivariate_normal(center, [[.02, -.008], [-.008, .015]], size=60)
        log_values[-1] += [3, 3]
        for x, y in np.exp(log_values):
            rows.append({"id": group, "io_type": io_type, "iops": x, "lat": y,
                         "n_jobs": 4 if group == "first" else 8, "raid": "4+1"})
    return pd.DataFrame(rows)


class CleanDatasetOutliersTest(unittest.TestCase):
    def test_group_local_reproducible_mask_preserves_entire_rows(self):
        frame = fixture()
        frame.index = np.arange(len(frame)) + 100
        cleaned, decisions, summary = clean_clouds(frame)
        second, repeated, _ = clean_clouds(frame)
        pd.testing.assert_frame_equal(cleaned, second)
        pd.testing.assert_frame_equal(decisions, repeated)
        pd.testing.assert_frame_equal(cleaned, frame.iloc[np.flatnonzero(decisions.keep)])
        self.assertFalse(decisions.keep.iloc[59])
        self.assertFalse(decisions.keep.iloc[119])
        self.assertTrue(summary.status.eq("fit").all())
        self.assertEqual(summary.rows_removed.tolist(), [3, 3])
        self.assertTrue(summary.rows_after.gt(50).all())

    def test_small_and_degenerate_clouds_retained_and_audited(self):
        frame = fixture().iloc[:10].copy()
        cleaned, decisions, summary = clean_clouds(frame)
        self.assertTrue(decisions.keep.all())
        self.assertEqual(summary.status.iloc[0], "skipped_small_cloud")
        frame = fixture().iloc[:60].copy()
        frame["lat"] = 100.
        cleaned, decisions, summary = clean_clouds(frame)
        self.assertEqual(len(cleaned), len(frame))
        self.assertEqual(summary.status.iloc[0], "skipped_rank_deficient")

    def test_zero_policy_is_explicit_without_dropping_other_rows(self):
        frame = fixture().iloc[:5].copy()
        frame.loc[frame.index[0], ["iops", "lat"]] = 0
        cleaned, decisions, summary = clean_clouds(frame)
        self.assertEqual(len(cleaned), 5)
        self.assertEqual(summary.zero_measurements_kept.iloc[0], 1)
        cleaned, decisions, summary = clean_clouds(frame, zero_policy="drop")
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(decisions.reason.iloc[0], "zero_policy")
        self.assertEqual(summary.zero_measurements_kept.iloc[0], 0)

    def test_invalid_measurements_and_mixed_context_rejected(self):
        frame = fixture()
        for bad in (-1, np.nan, np.inf):
            invalid = frame.copy()
            invalid.loc[0, "iops"] = bad
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                clean_clouds(invalid)
        frame.loc[0, "n_jobs"] = 999
        with self.assertRaisesRegex(ValueError, "multiple contexts"):
            clean_clouds(frame)
        with self.assertRaises(ValueError):
            clean_clouds(fixture(), contamination=0)

    def test_local_cli_writes_clean_copy_without_touching_inputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, test = root / "train.csv", root / "test.csv"
            fixture().to_csv(source, index=False)
            fixture().iloc[:2].to_csv(test, index=False)
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, test)]
            config = root / "datasets.json"
            config.write_text(json.dumps({"fixture": {"train_csv": str(source), "test_csv": str(test)}}))
            argv = ["clean_dataset_outliers.py", "--config", str(config), "--datasets", "fixture",
                    "--output-dir", str(root / "report"), "--cloud-examples", "2",
                    "--prepared-config", str(root / "report/datasets.json")]
            with patch.object(sys, "argv", argv):
                main()
            output = root / "report/fixture"
            decisions = pd.read_csv(output / "row_decisions.csv")
            actual = pd.read_csv(output / "train_clean.csv")
            expected = pd.read_csv(source).loc[decisions.keep].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected)
            prepared = dataset_paths(root / "report/datasets.json", "fixture")
            self.assertEqual(prepared["train_csv"], (output / "train_clean.csv").resolve())
            self.assertEqual(prepared["test_csv"], test.resolve())
            info = json.loads((output / "run_info.json").read_text())
            self.assertEqual(info["test_sha256"], hashes[1])
            self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, test)])
            for name in ("before_after.png", "before_after_log.png", "cloud_examples.png"):
                self.assertGreater((output / name).stat().st_size, 1000)
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                main()
        self.assertEqual(plt.get_fignums(), [])

    def test_preparation_rejects_overwriting_train_even_with_overwrite(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "fixture"
            directory.mkdir()
            source, test = directory / "train_clean.csv", root / "test.csv"
            fixture().to_csv(source, index=False)
            fixture().iloc[:2].to_csv(test, index=False)
            config = root / "datasets.json"
            config.write_text(json.dumps({"fixture": {"train_csv": str(source), "test_csv": str(test)}}))
            before = source.read_bytes()
            argv = ["clean_dataset_outliers.py", "--config", str(config), "--datasets", "fixture",
                    "--output-dir", str(root), "--overwrite"]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                main()
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
