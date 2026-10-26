from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import torch

from dss_imputer.diffusion import MixedDiffusion
from dss_imputer.model import MaskConditionedDenoiser
from dss_imputer.schema import TablePreprocessor
from scripts import train_dss_imputer as trainer
from scripts.eval_dss_imputer import main as eval_main
from scripts.plot_dss_imputer_loco_quality import main as plot_main


class ImputerContinuationTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        train = pd.DataFrame({
            "iops": [100.0 + 10*i for i in range(12)],
            "lat": [1e6 + 1e4*i for i in range(12)],
            "raid": ["r0" if i % 2 else "r1" for i in range(12)],
            "id": [f"train-{i}" for i in range(12)],
        })
        validation = train.iloc[:4].copy()
        validation["id"] = [f"test-{i}" for i in range(4)]
        self.train_csv, self.validation_csv = self.root / "train.csv", self.root / "test.csv"
        train.to_csv(self.train_csv, index=False)
        validation.to_csv(self.validation_csv, index=False)
        preprocessor = TablePreprocessor(["iops", "lat"], ["id"], ["iops", "lat"]).fit(train)
        denoiser = MaskConditionedDenoiser(2, preprocessor.categorical_sizes,
                                          d_model=8, layers=1, heads=1, dropout=0)
        self.diffusion = MixedDiffusion(denoiser, steps=2)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        state = {
            "epoch": 500, "validation_loss": 2.0,
            "model_config": denoiser.config(), "model_state": self.diffusion.state_dict(),
            "diffusion_config": self.diffusion.config(), "preprocessor": preprocessor.to_dict(),
            "run_config": {"train_csv": str(self.train_csv), "validation_csv": str(self.validation_csv)},
        }
        torch.save(state, self.model_dir / "last.pt")
        torch.save(state, self.model_dir / "best.pt")
        for name in ("history.csv", "run_config.json", "preprocessor.json", "generative_history.csv",
                     "generative_validation_real.csv", "generative_validation_loco_iops_latest.csv"):
            (self.model_dir / name).write_text("original contents")
        self.originals = {path.name: path.read_bytes() for path in self.model_dir.iterdir()}
        self.arguments = [
            "train_dss_imputer.py", "--train-csv", str(self.train_csv),
            "--validation-csv", str(self.validation_csv), "--output-dir", str(self.model_dir),
            "--resume-checkpoint", str(self.model_dir / "last.pt"),
            "--epochs", "2", "--batch-size", "1024", "--learning-rate", "5e-5",
            "--patience", "0", "--checkpoint-interval", "1",
            "--generative-validation-interval", "0", "--device", "cpu",
        ]

    def test_continuation_loads_weights_preserves_originals_and_supports_eval_plots(self):
        run_epoch = trainer.run_epoch
        captured = []

        def capture(*args, **kwargs):
            if not captured:
                captured.append({key: value.detach().clone() for key, value in args[0].state_dict().items()})
            return run_epoch(*args, **kwargs)

        with patch.object(sys, "argv", self.arguments), \
                patch.object(TablePreprocessor, "fit", side_effect=AssertionError("Must not refit")), \
                patch.object(trainer, "run_epoch", side_effect=capture):
            trainer.main()
        for name, contents in self.originals.items():
            self.assertEqual((self.model_dir / name).read_bytes(), contents, name)
        for name, tensor in self.diffusion.state_dict().items():
            self.assertTrue(torch.equal(tensor, captured[0][name]), name)
        continued = torch.load(self.model_dir / "last_continue.pt", map_location="cpu", weights_only=False)
        self.assertEqual(continued["epoch"], 502)
        self.assertEqual(continued["diffusion_config"]["steps"], 2)
        self.assertEqual(continued["model_config"], self.diffusion.denoiser.config())
        history = pd.read_csv(self.model_dir / "history_continue.csv")
        self.assertEqual(history.epoch.tolist(), [501, 502])
        self.assertTrue((self.model_dir / "epoch_0501_continue.pt").exists())
        self.assertTrue((self.model_dir / "best_continue.pt").exists())
        config = json.loads((self.model_dir / "run_config_continue.json").read_text())
        self.assertEqual(config["start_epoch"], 500)
        self.assertEqual(config["batch_size"], 1024)
        self.assertEqual(config["continuation_mode"], "weights_fresh_optimizer")
        self.assertEqual(config["architecture"], "transformer")
        self.assertEqual(config["log_continuous_cols"], ["iops", "lat"])
        self.assertTrue(any(not torch.equal(value, continued["model_state"][key])
                            for key, value in self.diffusion.state_dict().items()))
        results = self.root / "test_last_continue"
        with patch.object(sys, "argv", [
            "eval", "--data-csv", str(self.validation_csv), "--checkpoint", str(self.model_dir / "last_continue.pt"),
            "--output-dir", str(results), "--patterns", "loco", "--loco-cols", "iops",
            "--samples", "1", "--save-first-sample", "--device", "cpu",
        ]):
            eval_main()
        with patch.object(sys, "argv", [
            "plot", "--results-dir", str(results), "--plot-sample-size", "0",
            "--metric-sample-size", "0", "--pair-plot-kind", "scatter",
        ]):
            plot_main()
        self.assertTrue((results / "quality_plots/loco_iops/pairs/lat.png").exists())

    def test_existing_continuation_outputs_are_not_overwritten(self):
        path = self.model_dir / "last_continue.pt"
        path.write_bytes(b"already saved")
        with patch.object(sys, "argv", self.arguments):
            with self.assertRaisesRegex(ValueError, "already exists"):
                trainer.main()
        self.assertEqual(path.read_bytes(), b"already saved")

    def test_generative_artifacts_are_also_isolated(self):
        arguments = [*self.arguments, "--epochs", "1", "--generative-validation-interval", "1",
                     "--generative-validation-rows", "4", "--generative-dcr-rows", "4"]
        with patch.object(sys, "argv", arguments), \
                patch.object(trainer, "evaluate_generative_quality", return_value=({"overall_quality": 0.5}, [])):
            trainer.main()
        for name, contents in self.originals.items():
            self.assertEqual((self.model_dir / name).read_bytes(), contents, name)
        for name in ("generative_history_continue.csv", "generative_metric_errors_continue.csv",
                     "generative_validation_real_continue.csv",
                     "generative_validation_loco_iops_mask_continue.csv",
                     "generative_validation_loco_iops_latest_continue.csv"):
            self.assertTrue((self.model_dir / name).is_file(), name)

    def test_invalid_suffix_cannot_escape_output_directory(self):
        with patch.object(sys, "argv", [*self.arguments, "--artifact-suffix", "../bad"]):
            with self.assertRaisesRegex(ValueError, "artifact-suffix"):
                trainer.main()


if __name__ == "__main__":
    unittest.main()
