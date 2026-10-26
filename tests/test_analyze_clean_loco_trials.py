import unittest

import numpy as np
import pandas as pd

from scripts.analyze_clean_loco_trials import aggregate_trials, html_table


class CleanLocoTrialsTest(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            dict(dataset="cache", split="test", target="iops", model="hybrid",
                 trial=i, tv=value, c2st=.9, c2st_convergence_warning=False)
            for i, value in enumerate([.1, .2, .3, .4, .5])])

    def test_sample_std_and_all_trials(self):
        summary = aggregate_trials(self.fixture())
        row = summary[summary.metric == "tv"].iloc[0]
        self.assertAlmostEqual(row["mean"], .3)
        self.assertAlmostEqual(row["std"], np.std([.1, .2, .3, .4, .5], ddof=1))
        self.assertEqual(row.n, 5)
        self.assertTrue(row.complete)

    def test_missing_metrics_not_replaced_with_zero(self):
        frame = self.fixture()
        frame.loc[0, "tv"] = np.nan
        summary = aggregate_trials(frame)
        row = summary[summary.metric == "tv"].iloc[0]
        self.assertEqual(row.n, 4)
        self.assertFalse(row.complete)
        self.assertAlmostEqual(row["mean"], .35)
        self.assertIn("n=4", html_table(summary, ["tv"]))

    def test_duplicate_trial_rejected(self):
        frame = self.fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_trials(pd.concat([frame, frame.iloc[:1]]))

    def test_c2st_warning_is_not_ranked(self):
        frame = self.fixture()
        frame.loc[0, "c2st_convergence_warning"] = True
        summary = aggregate_trials(frame)
        self.assertNotIn("<u>", html_table(summary, ["c2st"]))
        self.assertIn("<u>", html_table(summary, ["tv"]))

    def test_mean_of_scores_not_score_of_mean_prediction(self):
        frame = self.fixture()
        frame["mse"] = [1, 4, 9, 16, 25]
        summary = aggregate_trials(frame)
        self.assertEqual(summary[summary.metric == "mse"].iloc[0]["mean"], 11)


if __name__ == "__main__":
    unittest.main()
