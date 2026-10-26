import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


class ExternalEvaluationSlurmTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.log = self.project / "jobs.jsonl"
        self.manifest = self.project / "manifest.tsv"
        self.bin = self.project / "bin"
        self.bin.mkdir()
        fake = self.bin / "sbatch"
        fake.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
p = Path(os.environ["MOCK_SBATCH_LOG"])
n = len(p.read_text().splitlines()) + 1 if p.exists() else 1
if os.environ.get("FAIL_SUBMISSION") == str(n): sys.exit(1)
keys = ["MODEL", "DATASET", "DATANAME", "CONDA_ENV", "ACTION", "PROTOCOL", "SPLIT",
        "OUTPUT_DIR", "PLOT_SAMPLE_SIZE", "PAIR_METRIC_SAMPLE_SIZE", "IMPUTE_CONDITION",
        "TABDIFF_EXP_NAME"]
with p.open("a") as f:
 f.write(json.dumps({"args": sys.argv[1:], "env": {k: os.environ.get(k) for k in keys}})+"\\n")
print(f"{20000+n};cluster")
''')
        fake.chmod(0o755)
        for dataset in ("hdd_sequential", "ssd_sequential", "ssd_random", "cache"):
            dataname = f"dss_{dataset}_uncond_all_20261015_iops"
            for model in ("tabsyn", "tabdiff"):
                repo = self.project / "external" / model
                data = repo / "data" / dataname
                data.mkdir(parents=True)
                for name in ("info.json", "train.csv", "test.csv"):
                    (data / name).touch()
                if model == "tabsyn":
                    paths = [repo / f"tabsyn/ckpt/{dataname}/model.pt",
                             repo / f"tabsyn/vae/ckpt/{dataname}/model.pt",
                             repo / f"tabsyn/vae/ckpt/{dataname}/train_z.npy"]
                else:
                    folder = repo / f"tabdiff/ckpt/{dataname}/tabdiff_{dataset}_uncond_all_20261015"
                    paths = [folder / "config.pkl", folder / "best_ema_model_1.pt"]
                for path in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
        self.env = {**os.environ, "PROJECT_DIR": str(self.project), "DATASETS": "cache",
                    "PHASE": "unconditional", "RUN_TAG": "uncond_all_20261015",
                    "MOCK_SBATCH_LOG": str(self.log), "JOB_MANIFEST": str(self.manifest),
                    "PATH": f"{self.bin}:{os.environ['PATH']}", "SUBMIT_DELAY": "0"}

    def run_submit(self, **overrides):
        return subprocess.run(["bash", str(ROOT / "slurm/submit_external_evaluation.sh"), "-A", "proj_1808"],
                              env={**self.env, **overrides}, capture_output=True, text=True)

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_unconditional_runs_only_native_sampling_metrics_and_plots(self):
        result = self.run_submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = self.records()
        self.assertEqual(len(jobs), 6)
        for start, model in ((0, "tabsyn"), (3, "tabdiff")):
            sample, metrics, plots = jobs[start:start + 3]
            self.assertEqual(sample["env"]["CONDA_ENV"], model)
            self.assertEqual(sample["env"]["PROTOCOL"], "unconditional")
            self.assertEqual(sample["env"]["SPLIT"], "train")
            self.assertIn(f"--dependency=afterok:{20001 + start}", metrics["args"])
            self.assertIn(f"--dependency=afterok:{20002 + start}", plots["args"])
            self.assertIn("--time=00:15:00", plots["args"])
            self.assertEqual(plots["env"]["PLOT_SAMPLE_SIZE"], "0")
            self.assertEqual(plots["env"]["PAIR_METRIC_SAMPLE_SIZE"], "0")
        self.assertTrue(self.manifest.is_file())

    def test_all_four_tables_have_8_unconditional_and_16_loco_sample_jobs(self):
        result = self.run_submit(PHASE="all", DATASETS="hdd_sequential ssd_sequential ssd_random cache")
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = self.records()
        self.assertEqual(len(jobs), 72)
        for start in range(0, len(jobs), 9):
            unconditional, train, test = jobs[start], jobs[start + 3], jobs[start + 6]
            self.assertIn(f"--dependency=afterok:{20002 + start}", train["args"])
            self.assertIn(f"--dependency=afterok:{20002 + start}", test["args"])
            self.assertEqual(train["env"]["SPLIT"], "train")
            self.assertEqual(test["env"]["SPLIT"], "test")
            self.assertEqual(train["env"]["IMPUTE_CONDITION"], "x_t")
            self.assertNotEqual(train["env"]["OUTPUT_DIR"], test["env"]["OUTPUT_DIR"])

    def test_loco_only_does_not_repeat_unconditional_sampling(self):
        result = self.run_submit(PHASE="loco")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.records()), 12)
        self.assertTrue(all(job["env"]["PROTOCOL"] == "loco" for job in self.records()))

    def test_clean_training_experiment_template_matches_preflight_and_sample(self):
        old = self.project / "external/tabdiff/tabdiff/ckpt/dss_cache_uncond_all_20261015_iops"
        (old / "tabdiff_cache_uncond_all_20261015").rename(
            old / "dss_tabdiff_dss_cache_uncond_all_20261015")
        result = self.run_submit(TABDIFF_EXP_NAME_TEMPLATE="dss_tabdiff_dss_{dataset}_{run_tag}")
        self.assertEqual(result.returncode, 0, result.stderr)
        for job in self.records():
            if job["env"]["MODEL"] == "tabdiff":
                self.assertEqual(job["env"]["TABDIFF_EXP_NAME"],
                                 "dss_tabdiff_dss_cache_uncond_all_20261015")

    def test_tabddpm_sampling_launcher_uses_clean_data_and_does_not_submit_other_models(self):
        scripts = self.project / "scripts"
        scripts.mkdir()
        (scripts / "training_data.py").write_text((ROOT / "scripts/training_data.py").read_text())
        train, test = self.project / "train_clean.csv", self.project / "test.csv"
        train.touch()
        test.touch()
        config = self.project / "datasets.json"
        config.write_text(json.dumps({"cache": {"train_csv": str(train), "test_csv": str(test)}}))
        result = subprocess.run(["bash", str(ROOT / "slurm/submit_sampling_and_reports.sh"),
                                 "-A", "proj_1808"], capture_output=True, text=True,
                                env={**self.env, "MODELS": "tabddpm", "DATASET_CONFIG": str(config),
                                     "DATASET_SUFFIX": "_clean_20261022_r2", "CONFIG_PYTHON": sys.executable})
        self.assertEqual(result.returncode, 0, result.stderr)
        sample, metrics = self.records()
        self.assertEqual(sample["args"][-1], "slurm/sample_tabddpm.sbatch")
        self.assertIn("--partition=rocky", sample["args"])
        self.assertIn("--dependency=afterok:20001", metrics["args"])
        for job in (sample, metrics):
            exports = next(arg for arg in job["args"] if arg.startswith("--export="))
            self.assertIn(f"REAL_TRAIN={train.resolve()}", exports)
            self.assertIn(f"REAL_TEST={test.resolve()}", exports)
            self.assertIn("dss_cache_clean_20261022_r2", exports)
        self.assertEqual(metrics["args"][-1], "slurm/metric_eval.sbatch")

    def test_dry_run_does_not_submit_or_create_manifest(self):
        result = self.run_submit(DRY_RUN="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("DRY RUN:"), 6)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.manifest.exists())

    def test_preflight_missing_last_dataset_prevents_partial_submission(self):
        (self.project / "external/tabdiff/data/dss_cache_uncond_all_20261015_iops/test.csv").unlink()
        result = self.run_submit(DATASETS="ssd_random cache")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_reports_reuses_samples_without_loading_or_requiring_weights(self):
        for model in ("tabsyn", "tabdiff"):
            path = self.project / f"reports/external_generators/uncond_all_20261015/{model}/cache/loco_train"
            path.mkdir(parents=True)
            (path / "run_info.json").write_text("{}")
        result = self.run_submit(PHASE="reports")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.records()), 4)
        self.assertTrue(all(job["env"]["ACTION"] in ("metrics", "plots") for job in self.records()))

    def test_failure_stops_submission_and_keeps_successful_job_ids(self):
        result = self.run_submit(FAIL_SUBMISSION="3")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.records()), 2)
        self.assertEqual(len(self.manifest.read_text().splitlines()), 3)

    def test_existing_trials_prevent_accidental_resampling(self):
        path = self.project / "reports/external_generators/uncond_all_20261015/tabdiff/cache/unconditional/unconditional"
        path.mkdir(parents=True)
        (path / "trial_000.csv").touch()
        result = self.run_submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PHASE=reports", result.stderr)
        self.assertFalse(self.log.exists())

    def test_new_job_headers_use_rocky_and_never_request_memory(self):
        for name in ("sample_external_generators.sbatch", "report_external_generators.sbatch"):
            text = (ROOT / "slurm" / name).read_text()
            self.assertIn("#SBATCH --partition=rocky", text)
            self.assertIn("#SBATCH --nodes=1", text)
            self.assertIn("#SBATCH --cpus-per-task=4", text)
            self.assertNotIn("--mem", text)


if __name__ == "__main__":
    unittest.main()
