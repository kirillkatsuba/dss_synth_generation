from __future__ import annotations

import argparse
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts import eval_external_generators as external
from scripts import tabdiff_impute, tabsyn_impute


class ExternalEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.data = self.repo / "data" / "toy"
        self.data.mkdir(parents=True)
        self.info = {"task_type": "regression", "column_names": ["iops", "lat", "io_type", "constant"],
                     "idx_name_mapping": {"0": "iops", "1": "lat", "2": "io_type", "3": "constant"},
                     "target_col_idx": [0], "num_col_idx": [1, 3], "cat_col_idx": [2]}
        (self.data / "info.json").write_text(json.dumps(self.info))
        rng = np.random.default_rng(2)
        self.train = pd.DataFrame({"iops": rng.uniform(1, 10, 120), "lat": rng.uniform(2, 20, 120),
                                   "io_type": ["read", "write"] * 60, "constant": 8})
        self.test = pd.DataFrame({"iops": rng.uniform(11, 20, 60), "lat": rng.uniform(20, 30, 60),
                                  "io_type": ["read", "write"] * 30, "constant": 8})
        self.train.to_csv(self.data / "train.csv", index=False)
        self.test.to_csv(self.data / "test.csv", index=False)
        self.args = argparse.Namespace(model="tabdiff", dataname="toy", repo=self.repo,
            exp_name="experiment", output_dir=self.root / "output", protocol="unconditional",
            split="test", trials=2, batch_size=64, steps=50, resample_rounds=1,
            impute_condition="x_t", seed=42, gpu=-1, metric_sample_size=120,
            dcr_sample_size=30, plot_sample_size=0, pair_metric_sample_size=0)

    def sample_unconditional(self):
        def generate(rows):
            self.assertEqual(rows, len(self.train))
            return self.train.copy()
        with patch.object(external, "unconditional_sampler", return_value=(generate, "weights.pt")):
            external.sample_results(self.args)

    def sample_loco(self, split="train"):
        self.args.protocol, self.args.split = "loco", split
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[command.index("--split") + 1], split)
            frame = pd.read_csv(self.data / f"{split}.csv")
            target = command[command.index("--mask-cols") + 1]
            directory = Path(command[command.index("--output-dir") + 1])
            directory.mkdir(parents=True)
            for trial in range(self.args.trials):
                sample = frame.copy()
                sample[target] = np.roll(sample[target], trial + 1)
                sample.to_csv(directory / f"trial_{trial:03d}.csv", index=False)
            (directory / "run_info.json").write_text(json.dumps({"checkpoint": "weights.pt"}))
        with patch.object(external.subprocess, "run", side_effect=fake_run):
            external.sample_results(self.args)
        return calls

    def test_unconditional_native_draws_use_train_size_without_conditioning(self):
        self.sample_unconditional()
        run = json.loads((self.args.output_dir / "run_info.json").read_text())
        self.assertEqual(run["split_label"], "train")
        self.assertEqual(run["rows"], 120)
        self.assertEqual(run["model_columns"], ["iops", "lat", "io_type"])
        self.assertFalse(run["cfg"])
        self.assertEqual(run["status"], "complete")
        saved = pd.read_csv(self.args.output_dir / "evaluation_truth.csv")
        pd.testing.assert_frame_equal(saved, pd.read_csv(self.data / "train.csv"))
        with self.assertRaises(FileExistsError):
            self.sample_unconditional()

    def test_loco_train_and_test_save_imputer_layout_without_constants(self):
        for split, rows in (("train", 120), ("test", 60)):
            self.args.output_dir = self.root / split
            calls = self.sample_loco(split)
            self.assertEqual(len(calls), 3)
            run = json.loads((self.args.output_dir / "run_info.json").read_text())
            self.assertEqual(run["rows"], rows)
            self.assertEqual(run["loco_columns"], ["iops", "lat", "io_type"])
            self.assertFalse(run["cfg"])
            for target in run["loco_columns"]:
                mask = pd.read_csv(self.args.output_dir / f"loco_{target}" / "target_mask.csv")
                self.assertEqual(list(mask), run["model_columns"])
                self.assertTrue(mask[target].all())
                self.assertFalse(mask.drop(columns=target).to_numpy().any())

    def test_tabdiff_native_sampler_does_not_use_imputation_or_builtin_mle(self):
        import torch
        diffusion = Mock()
        diffusion.sample_all.return_value = torch.ones(3, 4)
        with patch.object(tabdiff_impute, "resolve_checkpoint", return_value=Path("checkpoint.pt")), \
             patch.object(tabdiff_impute, "load_model", return_value=(diffusion, Mock(), Mock(), self.info)), \
             patch.object(tabdiff_impute, "recover_frame", return_value=self.train.iloc[:3]):
            generate, _ = external.unconditional_sampler(self.args)
            self.assertEqual(len(generate(3)), 3)
        diffusion.sample_all.assert_called_once_with(3, 64, keep_nan_samples=False)
        diffusion.sample_impute.assert_not_called()

    def fake_tabsyn_modules(self):
        import torch
        modules = {name: types.ModuleType(name) for name in (
            "tabsyn", "tabsyn.diffusion_utils", "tabsyn.model", "tabsyn.vae",
            "tabsyn.vae.model", "utils_train")}
        def fake_model(*args, **kwargs):
            model = Mock()
            model.to.return_value = model
            return model
        modules["tabsyn.model"].MLPDiffusion = fake_model
        modules["tabsyn.model"].Model = fake_model
        for name in ("Decoder_model", "Encoder_model", "Model_VAE"):
            setattr(modules["tabsyn.vae.model"], name, fake_model)
        self.native_sample = Mock(side_effect=lambda net, count, dim, **kwargs: torch.zeros(count, dim))
        modules["tabsyn.diffusion_utils"].sample = self.native_sample
        modules["utils_train"].preprocess = Mock(return_value=(
            (self.train[["iops", "lat", "constant"]].to_numpy(), self.test[["iops", "lat", "constant"]].to_numpy()),
            (np.zeros((120, 1), dtype=int), np.zeros((60, 1), dtype=int)),
            [2], 3, lambda values: values, lambda values: values,
        ))
        return modules

    @staticmethod
    def fake_decode(latent, *args):
        return pd.DataFrame({"iops": np.ones(len(latent)), "lat": np.ones(len(latent)),
                             "io_type": ["read"] * len(latent), "constant": 8})

    def test_tabsyn_uses_native_unconditional_sampler_with_bounded_batches(self):
        self.args.model = "tabsyn"
        modules = self.fake_tabsyn_modules()
        with patch.dict(sys.modules, modules), \
             patch.object(tabsyn_impute, "configure_imports"), \
             patch.object(tabsyn_impute, "load_state", return_value={}), \
             patch.object(tabsyn_impute, "decode_latent", side_effect=self.fake_decode), \
             patch.object(np, "load", return_value=np.zeros((120, 5, 4))):
            generate, _ = external.unconditional_sampler(self.args)
            self.assertEqual(len(generate(65)), 65)
        self.assertEqual([call.args[1] for call in self.native_sample.call_args_list], [64, 1])
        self.assertTrue(all(call.kwargs["num_steps"] == 50 for call in self.native_sample.call_args_list))

    def test_tabsyn_loco_reads_train_or_test_without_refitting_on_test(self):
        import torch
        for split, rows, expected in (("train", 120, self.train), ("test", 60, self.test)):
            modules = self.fake_tabsyn_modules()
            argv = ["tabsyn_impute.py", "--dataname", "toy", "--tabsyn-repo", str(self.repo),
                    "--output-dir", str(self.root / f"tabsyn_{split}"), "--mask-cols", "iops",
                    "--trials", "1", "--gpu", "-1", "--split", split]
            def encode(encoder, numerical, categorical, *args):
                self.assertEqual(len(numerical), rows)
                np.testing.assert_allclose(numerical[:, 1], expected.lat)
                return torch.zeros(rows, 16)
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                 patch.object(tabsyn_impute, "configure_imports"), \
                 patch.object(tabsyn_impute, "load_state", return_value={}), \
                 patch.object(tabsyn_impute, "encode_condition", side_effect=encode), \
                 patch.object(tabsyn_impute, "make_schedule", return_value=torch.tensor([1., 0.])), \
                 patch.object(tabsyn_impute, "sample_latent_batch", side_effect=lambda net, condition, *args: condition), \
                 patch.object(tabsyn_impute, "decode_latent", side_effect=self.fake_decode), \
                 patch.object(np, "load", return_value=np.zeros((120, 5, 4))):
                tabsyn_impute.main()
            sample = pd.read_csv(self.root / f"tabsyn_{split}" / "trial_000.csv")
            self.assertEqual(len(sample), rows)
            np.testing.assert_allclose(sample.lat, expected.lat)
            self.assertEqual(sample.io_type.tolist(), expected.io_type.tolist())

    def test_metrics_compare_unconditional_with_train_and_dcr_with_heldout(self):
        self.sample_unconditional()
        run = json.loads((self.args.output_dir / "run_info.json").read_text())
        with patch.object(external, "heldout_dcr", return_value=.25) as dcr:
            external.metric_results(self.args, run)
        self.assertEqual(dcr.call_count, 2)
        self.assertEqual(len(dcr.call_args.args[0]), 120)
        self.assertEqual(len(dcr.call_args.args[1]), 60)
        summary = pd.read_csv(self.args.output_dir / "generative_metrics_summary.csv")
        self.assertEqual(summary.loc[0, "dcr"], .25)
        self.assertAlmostEqual(summary.loc[0, "overall_quality"], 1)
        self.assertNotIn("r2", summary.columns)
        self.assertTrue((self.args.output_dir / "column_shapes_by_trial.csv").is_file())

    def test_loco_train_metrics_skip_invalid_dcr_and_verify_context(self):
        self.sample_loco("train")
        run = json.loads((self.args.output_dir / "run_info.json").read_text())
        with patch.object(external, "heldout_dcr") as dcr:
            external.metric_results(self.args, run)
        dcr.assert_not_called()
        self.assertEqual(pd.read_csv(self.args.output_dir / "context_integrity.csv").mismatches.sum(), 0)
        summary = pd.read_csv(self.args.output_dir / "generative_metrics_summary.csv")
        self.assertNotIn("dcr", summary)
        self.assertIn("target_overall_quality", summary)
        path = self.args.output_dir / "loco_iops" / "trial_000.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "lat"] += 1
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Known context changed"):
            external.metric_results(self.args, run)

    def test_tabdiff_main_uses_requested_transformed_split(self):
        import torch
        for split, rows in (("train", 120), ("test", 60)):
            train_dataset = Mock(X=torch.zeros(120, 4))
            test_dataset = Mock(X=torch.zeros(60, 4))
            args = ["tabdiff_impute.py", "--dataname", "toy", "--tabdiff-repo", str(self.repo),
                    "--output-dir", str(self.root / f"native_{split}"), "--mask-cols", "iops",
                    "--trials", "1", "--gpu", "-1", "--split", split]
            def copy_frame(*values):
                self.assertIs(values[2], train_dataset if split == "train" else test_dataset)
                return values[4].copy()
            with patch.object(sys, "argv", args), \
                 patch.object(tabdiff_impute, "resolve_checkpoint", return_value=Path("checkpoint.pt")), \
                 patch.object(tabdiff_impute, "load_model", return_value=(Mock(), train_dataset, test_dataset, self.info)), \
                 patch.object(tabdiff_impute, "impute_trial", side_effect=copy_frame):
                tabdiff_impute.main()
            saved = pd.read_csv(self.root / f"native_{split}" / "trial_000.csv")
            self.assertEqual(len(saved), rows)

    def test_unconditional_plots_reuse_stochastic_trial_not_point_estimate(self):
        self.sample_unconditional()
        run = json.loads((self.args.output_dir / "run_info.json").read_text())
        external.plot_results(self.args, run)
        self.assertTrue((self.args.output_dir / "quality_plots/iops_vs_lat.png").is_file())
        self.assertTrue((self.args.output_dir / "quality_plots/distributions/io_type/comparison.png").is_file())

    def test_loco_plots_reuse_existing_imputer_plotter(self):
        self.sample_loco("train")
        run = json.loads((self.args.output_dir / "run_info.json").read_text())
        external.plot_results(self.args, run)
        self.assertTrue((self.args.output_dir / "quality_plots/loco_iops/pairs/io_type.png").is_file())
        self.assertTrue((self.args.output_dir / "quality_plots/loco_io_type/distribution/comparison.png").is_file())
        self.assertEqual(len(pd.read_csv(self.args.output_dir / "quality_plots/pairwise_metrics.csv")), 6)


if __name__ == "__main__":
    unittest.main()
