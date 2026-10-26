import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from scripts import train_dss_imputer as trainer
from scripts import train_external_generators as external
from scripts.clean_dataset_outliers import main as clean_main
from scripts.training_data import ROOT, dataset_paths
from tests.test_clean_dataset_outliers import fixture


class TrainingDataConfigTest(unittest.TestCase):
    def test_existing_clean_config_is_portable_and_keeps_original_test_paths(self):
        configs = ROOT / "configs/dss_imputer"
        original = json.loads((configs / "datasets.json").read_text())
        cleaned = json.loads((configs / "datasets_clean.json").read_text())
        self.assertEqual(set(cleaned), set(original))
        for dataset, paths in cleaned.items():
            self.assertEqual(paths["test_csv"], original[dataset]["test_csv"])
            self.assertEqual(paths["train_csv"], f"reports/outlier_cleaning/{dataset}/train_clean.csv")
            self.assertFalse(any(Path(path).is_absolute() for path in paths.values()))

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.train, self.test = self.root / "train.csv", self.root / "test.csv"
        fixture().to_csv(self.train, index=False)
        test = fixture().iloc[:6].copy()
        test["id"] = "test-cloud"
        test.to_csv(self.test, index=False)
        self.config = self.root / "datasets.json"
        self.config.write_text(json.dumps({"toy": {"train_csv": str(self.train), "test_csv": str(self.test)}}))

    def parse(self, module, *extra):
        with patch.object(sys, "argv", ["train.py", "--dataset-config", str(self.config),
                                       "--dataset", "toy", *extra]):
            return module.parse_args()

    def test_config_resolves_both_trainers_and_rejects_partial_selection(self):
        args = self.parse(external, "--phase", "prepare")
        self.assertEqual(args.real_train, self.train)
        self.assertEqual(args.real_test, self.test)
        self.assertEqual(args.dataset_prefix, "dss_toy_clean")
        args = self.parse(trainer, "--output-dir", str(self.root / "model"))
        self.assertEqual(args.train_csv, self.train)
        self.assertEqual(args.validation_csv, self.test)
        self.assertEqual(args.patience, 0)
        self.assertEqual(self.parse(trainer, "--output-dir", str(self.root / "model"),
                                    "--patience", "7").patience, 7)
        for module in (external, trainer):
            argv = ["train.py", "--dataset-config", str(self.config)]
            if module is trainer:
                argv += ["--output-dir", str(self.root / "unused")]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                module.parse_args()

    def test_unknown_missing_and_identical_csvs_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            dataset_paths(self.config, "missing")
        self.train.unlink()
        with self.assertRaises(FileNotFoundError):
            dataset_paths(self.config, "toy")
        self.config.write_text(json.dumps({"toy": {"train_csv": str(self.test), "test_csv": str(self.test)}}))
        with self.assertRaisesRegex(ValueError, "different files"):
            dataset_paths(self.config, "toy")

    def test_external_preparation_uses_cleaned_rows_for_all_models(self):
        prepared = self.root / "clean/datasets.json"
        with patch.object(sys, "argv", ["clean.py", "--config", str(self.config), "--datasets", "toy",
                                       "--output-dir", str(prepared.parent), "--prepared-config", str(prepared),
                                       "--cloud-examples", "0"]):
            clean_main()
        self.config = prepared
        args = self.parse(external, "--phase", "prepare", "--dataset-prefix", "dss_toy_clean")
        repos = {name: {"path": self.root / name} for name in external.REPOS}
        with patch.object(external, "REPOS", repos), patch.object(external, "process_prepared_data"), \
             patch.object(external, "parse_args", return_value=args):
            external.main()
        expected = pd.read_csv(args.real_train)
        self.assertEqual(len(expected), 114)
        for model in ("tabsyn", "tabdiff"):
            folder = repos[model]["path"] / "data/dss_toy_clean_iops"
            actual = pd.read_csv(folder / "dss_toy_clean_iops.csv")
            spec = external.infer_spec(expected, "dss_toy_clean", "iops", ["id"])
            pd.testing.assert_frame_equal(actual, expected[spec.model_columns])
            pd.testing.assert_frame_equal(pd.read_csv(folder / "dss_toy_clean_iops_test.csv"),
                                          pd.read_csv(self.test)[spec.model_columns])
        info = json.loads((repos["tabddpm"]["path"] / "data/dss_toy_clean_iops/info.json").read_text())
        self.assertEqual(info["train_size"] + info["val_size"], len(expected))

    def test_dss_training_fits_cleaned_data_and_keeps_test_unchanged(self):
        prepared = self.root / "clean/datasets.json"
        before = self.test.read_bytes()
        with patch.object(sys, "argv", ["clean.py", "--config", str(self.config), "--datasets", "toy",
                                       "--output-dir", str(prepared.parent), "--prepared-config", str(prepared),
                                       "--cloud-examples", "0"]):
            clean_main()
        model = self.root / "model"
        with patch.object(sys, "argv", ["train.py", "--dataset-config", str(prepared), "--dataset", "toy",
                                       "--output-dir", str(model), "--epochs", "1", "--device", "cpu",
                                       "--batch-size", "128", "--d-model", "8", "--layers", "1", "--heads", "1",
                                       "--architecture", "transformer", "--diffusion-steps", "2",
                                       "--continuous-cols", "iops", "lat", "--generative-validation-interval", "0"]):
            trainer.main()
        config = json.loads((model / "run_config.json").read_text())
        self.assertEqual(config["train_rows"], 114)
        self.assertEqual(config["validation_rows"], 6)
        self.assertEqual(config["train_csv"], str(prepared.parent / "toy/train_clean.csv"))
        self.assertEqual(config["dataset_config"], str(prepared))
        preprocessor = json.loads((model / "preprocessor.json").read_text())
        clean = pd.read_csv(config["train_csv"])
        self.assertAlmostEqual(preprocessor["numerical_mean"]["iops"], clean.iops.mean())
        self.assertTrue((model / "last.pt").is_file())
        self.assertEqual(self.test.read_bytes(), before)

    def test_slurm_wrappers_forward_config_and_eval_resolves_clean_train(self):
        fake = self.root / "python"
        fake.write_text(f'#!/bin/bash\nif [[ "$1" == "scripts/training_data.py" ]]; then\nexec "{sys.executable}" "$@"\nfi\nprintf "%s\\n" "$@"\n')
        fake.chmod(0o755)
        conda_sh = self.root / "conda.sh"
        conda_sh.write_text('conda() { :; }\n')
        env = {**os.environ, "PROJECT_DIR": str(ROOT), "PYTHON_BIN": str(fake),
               "CONDA_SH": str(conda_sh), "CONDA_ENV": "fake", "DATASET_CONFIG": str(self.config),
               "DATASET": "hdd_sequential", "OUTPUT_DIR": str(self.root / "out"), "PLOT_AFTER_EVAL": "0"}
        # Use a known shell dataset key while resolving the custom CSV paths.
        self.config.write_text(json.dumps({"hdd_sequential": {"train_csv": str(self.train), "test_csv": str(self.test)}}))
        for name in ("prepare_external_generators", "train_dss_imputer", "train_tabddpm", "train_tabsyn", "train_tabdiff",
                     "sample_tabddpm", "sample_tabsyn", "sample_tabdiff"):
            result = subprocess.run(["bash", str(ROOT / f"slurm/{name}.sbatch")], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"--dataset-config\n{self.config}\n--dataset\nhdd_sequential", result.stdout)
            if name == "train_dss_imputer":
                self.assertIn(f"Train CSV: {self.train}", result.stdout)
        # Evaluate through the actual path resolver, mocking only inference.
        for split, expected in (("train", self.train), ("test", self.test)):
            result = subprocess.run(["bash", str(ROOT / "slurm/eval_dss_imputer.sbatch")],
                                    env={**env, "SPLIT": split}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"--data-csv\n{expected}\n", result.stdout)

    def test_cpu_preparation_job_runs_the_shared_cleaning_stage(self):
        conda_sh = self.root / "conda.sh"
        conda_sh.write_text('conda() { :; }\n')
        destination = self.root / "prepared"
        env = {**os.environ, "PROJECT_DIR": str(ROOT), "PYTHON_BIN": sys.executable,
               "CONDA_SH": str(conda_sh), "CONDA_ENV": "fake", "DATASETS": "toy",
               "SOURCE_CONFIG": str(self.config), "CLEAN_DATA_DIR": str(destination)}
        before = [path.read_bytes() for path in (self.train, self.test)]
        result = subprocess.run(["bash", str(ROOT / "slurm/prepare_training_data.sbatch")],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = dataset_paths(destination / "datasets.json", "toy")
        self.assertEqual(len(pd.read_csv(paths["train_csv"])), 114)
        self.assertEqual(paths["test_csv"], self.test)
        self.assertEqual(before, [path.read_bytes() for path in (self.train, self.test)])

    def test_training_wrapper_disables_early_stopping_unless_explicitly_requested(self):
        fake = self.root / "python"
        fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
        fake.chmod(0o755)
        conda_sh = self.root / "conda.sh"
        conda_sh.write_text('conda() { :; }\n')
        env = {**os.environ, "PROJECT_DIR": str(ROOT), "PYTHON_BIN": str(fake),
               "CONDA_SH": str(conda_sh), "CONDA_ENV": "fake", "DATASET": "cache",
               "OUTPUT_DIR": str(self.root / "model"), "EPOCHS": "1000"}
        for name in ("PATIENCE", "DATASET_CONFIG"):
            env.pop(name, None)
        for overrides, patience in (({}, "0"), ({"PATIENCE": "9"}, "9")):
            result = subprocess.run(["bash", str(ROOT / "slurm/train_dss_imputer.sbatch")],
                                    env={**env, **overrides}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"--patience\n{patience}\n", result.stdout)
            self.assertIn("--epochs\n1000\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
