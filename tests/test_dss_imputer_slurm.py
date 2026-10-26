import csv
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


class ImputerSlurmTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.project = Path(self.directory.name)
        for name, folder in [("ssd_sequential", "pools"), ("ssd_random", "pools"), ("cache", "cache")]:
            for split in ("train", "test"):
                path = self.project / "dataset" / folder / f"{split}_{name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("iops,lat,id\n1,2,a\n")
        self.manifest = self.project / "manifest.tsv"
        self.log = self.project / "sbatch.jsonl"
        bin_path = self.project / "bin"
        bin_path.mkdir()
        fake = bin_path / "sbatch"
        fake.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
log = Path(os.environ["MOCK_SBATCH_LOG"])
records = log.read_text().splitlines() if log.exists() else []
number = len(records)+1
if os.environ.get("FAIL_SUBMISSION") == str(number):
    sys.exit(1)
keys = ["DATASET", "OUTPUT_DIR", "CHECKPOINT", "SPLIT", "TRAINING_SUBSET", "PATTERNS",
        "LOCO_COLS", "EXCLUDE_LOCO_COLS", "SAMPLES", "SAVE_SAMPLES", "GENERATIVE_METRICS",
        "ARCHITECTURE", "DIFFUSION_STEPS", "MIN_MASK_RATE", "PATIENCE", "PAIR_PLOT_KIND",
        "CONDA_ENV", "RESULTS_DIR", "IOPS_LIMITS", "LAT_LIMITS", "LIMIT", "PLOT_AFTER_EVAL",
        "PLOT_SAMPLE_SIZE", "METRIC_SAMPLE_SIZE", "SAVE_FIRST_SAMPLE", "RESUME_CHECKPOINT",
        "ARTIFACT_SUFFIX", "BATCH_SIZE", "EPOCHS", "WEIGHT_DECAY"]
with log.open("a") as file:
    file.write(json.dumps({"args": sys.argv[1:], "env": {key: os.environ.get(key) for key in keys}})+"\\n")
print(f"{10000+number};mock-cluster")
''')
        fake.chmod(0o755)
        self.environment = {**os.environ, "PROJECT_DIR": str(self.project), "PHASE": "pipeline",
                            "MOCK_SBATCH_LOG": str(self.log), "JOB_MANIFEST": str(self.manifest),
                            "PATH": f"{bin_path}:{os.environ['PATH']}", "SUBMIT_DELAY": "0"}

    def run_pipeline(self, arguments=("-A", "proj_1808"), **overrides):
        return subprocess.run(["bash", str(ROOT / "slurm/submit_dss_imputer_all.sh"), *arguments],
                              env={**self.environment, **overrides}, capture_output=True, text=True)

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_pipeline_submits_three_training_and_dependent_best_last_eval_plots(self):
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        self.assertEqual(len(records), 15)
        for start in (0, 5, 10):
            train = records[start]
            self.assertEqual(train["env"]["DIFFUSION_STEPS"], "200")
            self.assertEqual(train["env"]["MIN_MASK_RATE"], "0.15")
            self.assertEqual(train["env"]["PATIENCE"], "0")
            self.assertEqual(train["env"]["RESUME_CHECKPOINT"], "")
            self.assertEqual(train["env"]["ARTIFACT_SUFFIX"], "")
            self.assertIn("--time=05:00:00", train["args"])
            self.assertTrue(any("hybrid_mcar15_log_t200_full_s42_train" in arg
                                for arg in train["args"]))
            for offset, checkpoint in ((1, "best"), (3, "last")):
                evaluation = records[start+offset]
                self.assertIn(f"--dependency=afterok:{10001+start}", evaluation["args"])
                self.assertEqual(evaluation["env"]["SPLIT"], "test")
                self.assertEqual(evaluation["env"]["PATTERNS"], "loco")
                self.assertEqual(evaluation["env"]["LOCO_COLS"], "")
                self.assertEqual(evaluation["env"]["GENERATIVE_METRICS"], "1")
                self.assertEqual(evaluation["env"]["SAVE_SAMPLES"], "1")
                self.assertIn("--time=05:00:00", evaluation["args"])
                self.assertTrue(evaluation["env"]["CHECKPOINT"].endswith(f"/{checkpoint}.pt"))
                plot = records[start+offset+1]
                self.assertIn(f"--dependency=afterok:{10001+start+offset}", plot["args"])
                self.assertEqual(plot["env"]["PAIR_PLOT_KIND"], "scatter")
                self.assertEqual(plot["env"]["IOPS_LIMITS"], "")
                self.assertIn("--time=00:15:00", plot["args"])
                self.assertEqual(plot["env"]["PLOT_SAMPLE_SIZE"], "0")
                self.assertEqual(plot["env"]["METRIC_SAMPLE_SIZE"], "0")
        with self.manifest.open() as file:
            saved = list(csv.DictReader(file, delimiter="\t"))
        self.assertEqual(len(saved), 15)
        self.assertEqual(len({row["job_id"] for row in saved}), 15)

    def test_fresh_cache_1000_epochs_400_steps_ignores_inherited_resume(self):
        result = self.run_pipeline(DATASETS="cache", EPOCHS="1000", DIFFUSION_STEPS="400",
                                   TRAIN_BATCH_SIZE="512", RUN_TAG="hybrid_mcar15_log_t400_e1000_full_s42",
                                   RESUME_CHECKPOINT="old/last.pt", ARTIFACT_SUFFIX="continue")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        self.assertEqual(len(records), 5)
        train = records[0]
        self.assertEqual(train["env"]["EPOCHS"], "1000")
        self.assertEqual(train["env"]["DIFFUSION_STEPS"], "400")
        self.assertEqual(train["env"]["BATCH_SIZE"], "512")
        self.assertEqual(train["env"]["RESUME_CHECKPOINT"], "")
        self.assertEqual(train["env"]["ARTIFACT_SUFFIX"], "")
        self.assertTrue(train["env"]["OUTPUT_DIR"].endswith("cache_hybrid_mcar15_log_t400_e1000_full_s42"))

    def completed_models(self):
        for dataset in ("ssd_sequential", "ssd_random", "cache"):
            run = f"{dataset}_hybrid_mcar15_log_t200_full_s42"
            models = self.project / "trained_models/dss_imputer" / run
            models.mkdir(parents=True)
            for checkpoint in ("best", "last"):
                (models / f"{checkpoint}.pt").touch()
                results = self.project / "reports/dss_imputer" / run / f"test_{checkpoint}"
                results.mkdir(parents=True)
                (results / "run_info.json").write_text("{}")
                (results / "evaluation_truth.csv").write_text("iops,lat\n1,2\n")

    def test_reports_reuses_test_samples_and_evaluates_full_train(self):
        self.completed_models()
        result = self.run_pipeline(PHASE="reports")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        self.assertEqual(len(records), 18)
        for start in range(0, 18, 3):
            test_plot, train_eval, train_plot = records[start:start+3]
            for plot in (test_plot, train_plot):
                self.assertEqual(plot["args"][-1], "slurm/plot_dss_imputer_loco_quality.sbatch")
                self.assertEqual(plot["env"]["PLOT_SAMPLE_SIZE"], "0")
                self.assertEqual(plot["env"]["METRIC_SAMPLE_SIZE"], "0")
                self.assertTrue(plot["env"]["OUTPUT_DIR"].endswith("quality_plots_full"))
            self.assertFalse(any(arg.startswith("--dependency") for arg in test_plot["args"]))
            self.assertEqual(train_eval["args"][-1], "slurm/eval_dss_imputer.sbatch")
            self.assertEqual(train_eval["env"]["SPLIT"], "train")
            self.assertEqual(train_eval["env"]["TRAINING_SUBSET"], "all")
            self.assertEqual(train_eval["env"]["PATTERNS"], "loco")
            self.assertEqual(train_eval["env"]["LIMIT"], "")
            self.assertEqual(train_eval["env"]["LOCO_COLS"], "")
            self.assertEqual(train_eval["env"]["SAVE_SAMPLES"], "1")
            self.assertEqual(train_eval["env"]["SAVE_FIRST_SAMPLE"], "1")
            self.assertEqual(train_eval["env"]["PLOT_AFTER_EVAL"], "0")
            self.assertEqual(train_eval["env"]["METRIC_SAMPLE_SIZE"], "5000")
            self.assertIn(f"--dependency=afterok:{10002+start}", train_plot["args"])
        self.assertFalse(any(record["args"][-1] == "slurm/train_dss_imputer.sbatch"
                             for record in records))

    def test_reports_missing_result_prevents_all_submissions(self):
        self.completed_models()
        missing = self.project / "reports/dss_imputer/cache_hybrid_mcar15_log_t200_full_s42/test_last/run_info.json"
        missing.unlink()
        result = self.run_pipeline(PHASE="reports")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_reports_dry_run_never_submits(self):
        self.completed_models()
        result = self.run_pipeline(PHASE="reports", DRY_RUN="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("DRY RUN:"), 18)
        self.assertFalse(self.log.exists())

    def test_continue_submits_cache_training_then_last_continue_eval_and_plots(self):
        self.completed_models()
        result = self.run_pipeline(PHASE="continue", CONTINUE_EPOCHS="200")
        self.assertEqual(result.returncode, 0, result.stderr)
        train, evaluation, plot = self.records()
        self.assertEqual(train["env"]["DATASET"], "cache")
        self.assertTrue(train["env"]["RESUME_CHECKPOINT"].endswith("/last.pt"))
        self.assertEqual(train["env"]["BATCH_SIZE"], "512")
        self.assertEqual(train["env"]["EPOCHS"], "200")
        self.assertEqual(train["env"]["ARTIFACT_SUFFIX"], "continue")
        self.assertEqual(train["env"]["WEIGHT_DECAY"], "1e-4")
        self.assertIn("--dependency=afterok:10001", evaluation["args"])
        self.assertTrue(evaluation["env"]["CHECKPOINT"].endswith("/last_continue.pt"))
        self.assertEqual(evaluation["env"]["PATTERNS"], "loco")
        self.assertEqual(evaluation["env"]["LOCO_COLS"], "")
        self.assertIn("--dependency=afterok:10002", plot["args"])
        self.assertIn("--time=00:15:00", plot["args"])
        self.assertEqual(plot["env"]["PLOT_SAMPLE_SIZE"], "0")
        self.assertEqual(plot["env"]["METRIC_SAMPLE_SIZE"], "0")

    def test_continue_existing_outputs_prevent_submission(self):
        self.completed_models()
        path = self.project / "trained_models/dss_imputer/cache_hybrid_mcar15_log_t200_full_s42/last_continue.pt"
        path.touch()
        result = self.run_pipeline(PHASE="continue")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_continue_dry_run_and_optional_best_continue(self):
        self.completed_models()
        result = self.run_pipeline(PHASE="continue", DRY_RUN="1", CHECKPOINTS="best_continue last_continue")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("DRY RUN:"), 5)
        self.assertFalse(self.log.exists())

    def test_time_limits_can_be_overridden(self):
        result = self.run_pipeline(TRAIN_TIME="06:00:00", EVAL_TIME="03:00:00",
                                   PLOT_TIME="01:00:00")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        for start in (0, 5, 10):
            self.assertIn("--time=06:00:00", records[start]["args"])
            for offset in (1, 3):
                self.assertIn("--time=03:00:00", records[start+offset]["args"])
                self.assertIn("--time=01:00:00", records[start+offset+1]["args"])

    def test_sbatch_headers_explicitly_select_partition_and_resources(self):
        for filename in ("train_dss_imputer.sbatch", "eval_dss_imputer.sbatch",
                         "impute_dss_imputer.sbatch", "plot_dss_imputer.sbatch",
                         "plot_dss_imputer_loco_quality.sbatch"):
            with self.subTest(filename=filename):
                script = (ROOT / "slurm" / filename).read_text()
                self.assertTrue(script.startswith("#!/bin/bash\n"))
                self.assertIn("#SBATCH --partition=rocky\n", script)
                self.assertIn("#SBATCH --nodes=1\n", script)
                self.assertIn("#SBATCH --cpus-per-task=4\n", script)
                self.assertIn("#SBATCH --output=slurm/logs/%x-%j.out\n", script)
                self.assertIn("#SBATCH --error=slurm/logs/%x-%j.err\n", script)
                if filename.startswith("plot_"):
                    self.assertNotIn("#SBATCH --gpus=1", script)
                else:
                    self.assertIn("#SBATCH --gpus=1\n", script)

    def test_slurm_files_do_not_request_memory(self):
        for path in (ROOT / "slurm").iterdir():
            if path.suffix not in {".sbatch", ".sh"}:
                continue
            with self.subTest(filename=path.name):
                script = path.read_text()
                self.assertNotRegex(script, r"--mem(?:[=\s]|-per-(?:cpu|gpu))")
                self.assertNotIn("SBATCH_MEM_", script)

    def test_dry_run_without_sbatch_arguments_never_submits(self):
        result = self.run_pipeline(arguments=(), DRY_RUN="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())
        self.assertEqual(result.stdout.count("DRY RUN:"), 15)

    def test_missing_data_prevents_any_submission(self):
        (self.project / "dataset/cache/test_cache.csv").unlink()
        result = self.run_pipeline()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_submission_failure_stops_and_preserves_successful_job_ids(self):
        result = self.run_pipeline(FAIL_SUBMISSION="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.records()), 1)
        with self.manifest.open() as file:
            saved = list(csv.DictReader(file, delimiter="\t"))
        self.assertEqual(saved[0]["job_id"], "10001")


if __name__ == "__main__":
    unittest.main()
