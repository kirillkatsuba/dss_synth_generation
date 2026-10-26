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
from dss_imputer.splits import split_indices
from scripts.eval_dss_imputer import main as eval_main, select_training_subset
from scripts.plot_dss_imputer_evaluation import main as plot_main
from scripts.plot_dss_imputer_loco_quality import main as quality_main
from scripts.train_dss_imputer import (
    main as train_main,
    prepare_generative_validation,
)


class DssImputerEvalPipelineTest(unittest.TestCase):
    def test_external_validation_trains_on_every_train_row(self) -> None:
        train = pd.DataFrame({
            "iops": [100.0 + 10 * i for i in range(12)],
            "lat": [1e6 + 1e4 * i for i in range(12)],
            "raid": ["r0" if i % 2 else "r1" for i in range(12)],
            "id": [f"train-{i}" for i in range(12)],
        })
        validation = pd.DataFrame({
            "iops": [120.0, 140.0, 160.0, 180.0],
            "lat": [1.02e6, 1.04e6, 1.06e6, 1.08e6],
            "raid": ["r0", "r1", "r0", "r1"],
            "id": [f"validation-{i}" for i in range(4)],
        })
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_csv, validation_csv = root / "train.csv", root / "validation.csv"
            train.to_csv(train_csv, index=False)
            validation.to_csv(validation_csv, index=False)
            model_dir = root / "model"
            with patch.object(sys, "argv", [
                "train_dss_imputer.py", "--train-csv", str(train_csv),
                "--validation-csv", str(validation_csv), "--output-dir", str(model_dir),
                "--continuous-cols", "iops", "lat", "--epochs", "1",
                "--checkpoint-interval", "1",
                "--batch-size", "4", "--diffusion-steps", "2", "--d-model", "8",
                "--layers", "1", "--heads", "1", "--architecture", "transformer",
                "--generative-validation-interval", "1",
                "--generative-validation-rows", "4",
                "--generative-validation-samples", "1",
                "--generative-validation-loco-cols", "raid",
                "--generative-validation-batch-size", "4",
                "--generative-dcr-rows", "4", "--device", "cpu",
            ]):
                train_main()

            config = json.loads((model_dir / "run_config.json").read_text())
            self.assertEqual(config["split_method"], "external_csv")
            self.assertEqual(config["validation_csv"], str(validation_csv))
            self.assertEqual(config["train_rows"], len(train))
            self.assertEqual(config["validation_rows"], len(validation))
            self.assertEqual(config["checkpoint_interval"], 1)
            history = pd.read_csv(model_dir / "history.csv")
            self.assertEqual(len(history), 1)
            generative = pd.read_csv(model_dir / "generative_history.csv")
            loco = generative[generative.scenario == "loco_raid"].iloc[0]
            self.assertEqual(loco["target_raid_pair_valid_count"], 2)
            self.assertEqual(loco["target_raid_rows"], 4)
            self.assertIn("metric_protocol", generative)
            self.assertTrue((model_dir / "best.pt").is_file())
            self.assertTrue((model_dir / "last.pt").is_file())
            self.assertTrue((model_dir / "epoch_0001.pt").is_file())
            checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
            periodic = torch.load(model_dir / "epoch_0001.pt", map_location="cpu", weights_only=False)
            self.assertEqual(periodic["epoch"], checkpoint["epoch"])
            self.assertEqual(periodic["model_config"], checkpoint["model_config"])
            self.assertAlmostEqual(
                checkpoint["preprocessor"]["numerical_mean"]["iops"],
                train.iops.mean(),
            )
            preprocessor = TablePreprocessor.from_dict(checkpoint["preprocessor"])
            real, scenarios = prepare_generative_validation(
                validation,
                preprocessor.transform(validation),
                preprocessor,
                rows=2,
                mask_rate=0.3,
                loco_columns=["iops"],
                min_context=1,
                seed=42,
            )
            self.assertEqual(len(real), 2)
            self.assertTrue(set(real.id).issubset(set(validation.id)))
            self.assertTrue(scenarios["loco_iops"][0].iops.isna().all())
            fit, _ = select_training_subset(train, checkpoint, train_csv, "fit")
            held_out, _ = select_training_subset(
                validation, checkpoint, validation_csv, "validation"
            )
            pd.testing.assert_frame_equal(fit, train)
            pd.testing.assert_frame_equal(held_out, validation)
            with self.assertRaisesRegex(ValueError, "checkpoint's CSV"):
                select_training_subset(train, checkpoint, train_csv, "validation")
            results_dir = root / "validation_results"
            with patch.object(sys, "argv", [
                "eval_dss_imputer.py", "--data-csv", str(validation_csv),
                "--checkpoint", str(model_dir / "best.pt"),
                "--output-dir", str(results_dir),
                "--training-subset", "validation", "--patterns", "loco",
                "--loco-cols", "raid", "--samples", "1", "--batch-size", "4",
                "--generative-metrics", "--metric-sample-size", "4",
                "--dcr-sample-size", "4",
                "--device", "cpu",
            ]):
                eval_main()
            run_info = json.loads((results_dir / "run_info.json").read_text())
            self.assertEqual(run_info["rows"], len(validation))
            self.assertTrue(run_info["used_for_checkpoint_selection"])
            self.assertTrue(
                (results_dir / "deterministic_metrics_by_trial.csv").is_file()
            )
            self.assertTrue(
                (results_dir / "deterministic_metrics_trial_summary.csv").is_file()
            )
            self.assertTrue(
                (results_dir / "generative_metrics_by_trial.csv").is_file()
            )
            self.assertTrue(
                (results_dir / "generative_metrics_summary.csv").is_file()
            )
            self.assertTrue((results_dir / "loco_raid" / "target_mask.csv").is_file())
            metrics = pd.read_csv(results_dir / "generative_metrics_by_trial.csv")
            self.assertEqual(metrics.target_raid_pair_valid_count.iloc[0], 2)
            pd.testing.assert_frame_equal(
                pd.read_csv(results_dir / "evaluation_truth.csv"), validation
            )

    def test_external_validation_rejects_overlapping_groups(self) -> None:
        frame = pd.DataFrame({
            "iops": [1.0, 2.0], "raid": ["r0", "r1"], "id": ["same", "other"]
        })
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_csv, validation_csv = root / "train.csv", root / "validation.csv"
            frame.to_csv(train_csv, index=False)
            frame.iloc[:1].to_csv(validation_csv, index=False)
            with patch.object(sys, "argv", [
                "train_dss_imputer.py", "--train-csv", str(train_csv),
                "--validation-csv", str(validation_csv),
                "--output-dir", str(root / "model"),
                "--continuous-cols", "iops", "--generative-validation-interval", "0",
            ]):
                with self.assertRaisesRegex(ValueError, "share 1 'id' groups"):
                    train_main()

    def test_validation_loco_uses_checkpoint_split_and_plots_all_target_types(self) -> None:
        frame = pd.DataFrame(
            {
                "iops": [100.0 + i * 10 for i in range(20)],
                "lat": [1e6 + i * 1e4 for i in range(20)],
                "iodepth": [1 + i % 3 for i in range(20)],
                "io_type": ["read" if i % 2 else "write" for i in range(20)],
                "raid": ["r0" if i % 3 else "r1" for i in range(20)],
                "id": [f"group-{i // 2}" for i in range(20)],
            }
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_csv = root / "train.csv"
            checkpoint_path = root / "last.pt"
            results_dir = root / "validation_loco"
            plots_dir = results_dir / "plots"
            frame.to_csv(train_csv, index=False)

            fit_indices, validation_indices, _ = split_indices(frame, 0.2, "id", 42)
            preprocessor = TablePreprocessor(
                ["iops", "lat", "iodepth"], ["id"]
            ).fit(frame.iloc[fit_indices])
            denoiser = MaskConditionedDenoiser(
                len(preprocessor.numerical_columns),
                preprocessor.categorical_sizes,
                d_model=8,
                layers=1,
                heads=1,
                dropout=0,
                ff_multiplier=2,
            )
            diffusion = MixedDiffusion(denoiser, steps=2)
            torch.save(
                {
                    "epoch": 1,
                    "model_state": diffusion.state_dict(),
                    "model_config": denoiser.config(),
                    "diffusion_config": diffusion.config(),
                    "preprocessor": preprocessor.to_dict(),
                    "run_config": {
                        "train_csv": str(train_csv),
                        "max_rows": None,
                        "validation_fraction": 0.2,
                        "group_column": "id",
                        "seed": 42,
                    },
                },
                checkpoint_path,
            )

            with patch.object(
                sys,
                "argv",
                [
                    "eval_dss_imputer.py",
                    "--data-csv", str(train_csv),
                    "--checkpoint", str(checkpoint_path),
                    "--output-dir", str(results_dir),
                    "--training-subset", "validation",
                    "--patterns", "loco",
                    "--exclude-loco-cols", "iops", "lat",
                    "--samples", "1",
                    "--save-first-sample",
                    "--batch-size", "8",
                    "--device", "cpu",
                ],
            ):
                eval_main()

            selected = pd.read_csv(results_dir / "evaluation_truth.csv")
            self.assertEqual(
                selected.id.tolist(), frame.iloc[validation_indices].id.tolist()
            )
            self.assertEqual(
                set(pd.read_csv(results_dir / "metrics_by_column.csv").column),
                {"iodepth", "io_type", "raid"},
            )
            run_info = json.loads((results_dir / "run_info.json").read_text())
            self.assertEqual(run_info["training_subset"], "validation")
            self.assertTrue((results_dir / "loco_iodepth" / "trial_000.csv").is_file())
            self.assertTrue((results_dir / "metrics_first_sample.csv").is_file())

            with patch.object(
                sys,
                "argv",
                [
                    "plot_dss_imputer_evaluation.py",
                    "--results-dir", str(results_dir),
                    "--data-csv", str(train_csv),
                    "--output-dir", str(plots_dir),
                    "--density-scale", "linear",
                ],
            ):
                plot_main()

            self.assertTrue((plots_dir / "loco_iodepth" / "row_comparison.png").is_file())
            self.assertTrue(
                (plots_dir / "loco_iodepth" / "prediction_vs_truth.png").is_file()
            )
            self.assertTrue((plots_dir / "loco_io_type" / "row_comparison.png").is_file())
            self.assertTrue((plots_dir / "loco_io_type" / "confusion_matrix.png").is_file())
            self.assertEqual(
                json.loads((plots_dir / "run_info.json").read_text())["rows"],
                len(validation_indices),
            )

            quality_dir = results_dir / "quality_plots"
            with patch.object(
                sys,
                "argv",
                ["plot_dss_imputer_loco_quality.py", "--results-dir", str(results_dir),
                 "--output-dir", str(quality_dir), "--plot-sample-size", "20",
                 "--metric-sample-size", "20"],
            ):
                quality_main()
            pairs = pd.read_csv(quality_dir / "pairwise_metrics.csv")
            self.assertEqual(set(pairs.pair_type), {"num-num", "num-cat", "cat-num", "cat-cat"})
            self.assertTrue((quality_dir / "loco_iodepth" / "distribution" / "comparison.png").is_file())
            self.assertTrue((quality_dir / "loco_io_type" / "pairs" / "raid.png").is_file())
            self.assertTrue((quality_dir / "loco_raid" / "pairs" / "iodepth.png").is_file())

            test_csv = root / "test.csv"
            test_frame = frame.iloc[10:].reset_index(drop=True)
            test_frame.to_csv(test_csv, index=False)
            test_results = root / "test_loco"
            with patch.object(
                sys,
                "argv",
                ["eval_dss_imputer.py", "--data-csv", str(test_csv),
                 "--checkpoint", str(checkpoint_path), "--output-dir", str(test_results),
                 "--training-subset", "all", "--split-label", "test",
                 "--patterns", "loco", "--loco-cols", "io_type", "--samples", "1",
                 "--save-first-sample", "--batch-size", "8", "--device", "cpu"],
            ):
                eval_main()
            saved_test = pd.read_csv(test_results / "evaluation_truth.csv")
            pd.testing.assert_frame_equal(saved_test, test_frame)
            self.assertEqual(
                json.loads((test_results / "run_info.json").read_text())["split_label"],
                "test",
            )
            with patch.object(
                sys,
                "argv",
                ["plot_dss_imputer_loco_quality.py", "--results-dir", str(test_results),
                 "--plot-sample-size", "20", "--metric-sample-size", "20"],
            ):
                quality_main()
            self.assertTrue(
                (test_results / "quality_plots" / "loco_io_type" / "distribution" / "comparison.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
