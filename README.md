# DSS Synthetic Generation and Imputation

Research code for generating storage benchmark tables and reconstructing missing
values from observed context. The main model, **DSS Imputer**, uses mixed numerical
and categorical diffusion with Transformer, Hybrid (Transformer -> MLP ->
Transformer), and MLP denoisers. Baseline integrations cover
[TabDiff](https://github.com/minkaixu/tabdiff),
[TabSyn](https://github.com/amazon-science/tabsyn),
[TabDDPM](https://github.com/yandex-research/tab-ddpm), and SDV CTGAN.

The repository includes training, conditional sampling, repeated-trial evaluation,
distribution/scatter plots, outlier cleaning, and SLURM workflows for four datasets.
It is research software, not a pretrained model distribution.

## Repository Layout

```text
dss_imputer/       Model, diffusion, masking, preprocessing, inference and metrics
scripts/           Training, sampling, evaluation, plotting and analysis entry points
slurm/             Cluster jobs and dependency-aware submission scripts
configs/           Dataset path mappings
docs/              Detailed experiment workflows and metric protocols
tests/             Unit, integration and submission-script tests
```

Datasets, checkpoints, generated CSVs, reports, logs, credentials, and cloned
upstream repositories are excluded from new commits. Older Git history may still
contain experiment artifacts; removing them from the current tree does not erase
history. Supply the datasets separately before running any experiment.

## Installation

Run commands from the repository root. The core code requires Python 3.10 or later.
For a local CPU environment:

```bash
git clone https://github.com/kirillkatsuba/dss_synth_generation.git
cd dss_synth_generation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` records the core versions used for local verification, not a
portable lockfile for every accelerator. On a GPU server, select a PyTorch build
compatible with its driver. Install `requirements-optional.txt` for the legacy
SDV/CTGAN and XGBoost workflows, or `requirements-dev.txt` for tests.

**Use separate environments for the upstream models.** Install each model's own
dependencies in its cloned repository; do not overlay this project's pinned
requirements onto all upstream environments. Cluster examples use `tabdiff` for
DSS Imputer/TabDiff, `tabsyn` for TabSyn, and `tddpm` for TabDDPM. Environment names
are configurable. ClearML is optional and disabled unless explicitly configured;
keep credentials outside the repository.

## Data and Evaluation Protocol

Default paths are defined in [datasets.json](configs/dss_imputer/datasets.json):

| Dataset key | Train CSV | Test CSV |
| --- | --- | --- |
| `hdd_sequential` | `dataset/pools/train_hdd_sequential.csv` | `dataset/pools/test_hdd_sequential.csv` |
| `ssd_sequential` | `dataset/pools/train_ssd_sequential.csv` | `dataset/pools/test_ssd_sequential.csv` |
| `ssd_random` | `dataset/pools/train_ssd_random.csv` | `dataset/pools/test_ssd_random.csv` |
| `cache` | `dataset/cache/train_cache.csv` | `dataset/cache/test_cache.csv` |

The current DSS defaults model `iops`, `lat`, `n_jobs`, and `iodepth` as numerical.
Other nonconstant columns, including `block_size` and `n_disks`, are categorical.
`id` is excluded from the model and used to check group separation. Constants are
restored deterministically. Types and preprocessing are fitted on training data
and stored in the checkpoint; inspect the saved schema when comparing older runs.

**The current cluster workflow trains on the full training CSV and uses the test
CSV for validation and `best` checkpoint selection.** Consequently, results on
that test CSV are not an untouched holdout estimate. For an independent final
evaluation, pass explicit `--train-csv` and `--validation-csv` paths without
`--dataset-config`, and reserve the test CSV for evaluation only. The shared
dataset config overrides explicit paths and uses its `test_csv` for validation.
Group overlap between train and validation is
rejected by default. Do not describe validation-guided checkpoint selection as
independent testing.

### Optional Outlier Cleaning

```bash
python scripts/clean_dataset_outliers.py \
  --output-dir dataset/cleaned \
  --prepared-config dataset/cleaned/datasets.json
```

This fits an `EllipticEnvelope` separately within each `id + io_type` cloud in
`log(iops + epsilon), log(lat + epsilon)` coordinates. It filters **training rows
only** and preserves the original test tables. Default contamination is 0.05;
small or degenerate clouds are retained and audited. This is an explicit trimming
policy, not proof that every rejected point is an anomaly.

Use the same generated config for every model to compare them on identical data.
See [cleaning and training workflows](docs/cloud_outlier_cleaning.md) for audits,
cluster preparation, baseline dataset suffixes, and reuse of cleaned CSVs.

## How DSS Imputer Works

1. Fit numerical scaling, optional `log(x + epsilon)`, and categorical vocabularies
   on training data. Sample an artificial target mask over observed nonconstant
   cells; naturally missing cells are not supervised as ground truth.
2. Noise only the selected targets: Gaussian DDPM noise for numerical values and
   an absorbing `[MASK]` process for categories. Observed context remains clean.
3. Encode each column as a token. Numerical tokens use column-specific affine
   projections; categorical tokens use disjoint embedding ranges. A learned status
   embedding distinguishes context from targets. Explicit column/type embeddings
   are currently disabled in the implementation.
4. The Hybrid backbone applies a Transformer, a flattened time-conditioned MLP,
   then a second Transformer. Output heads predict numerical noise and one
   categorical distribution per categorical column. Transformer-only and MLP
   backbones are also available.
5. Optimize numerical noise MSE and categorical cross-entropy on eligible targets;
   categorical supervision applies when a target is in the absorbing state.
   Reverse diffusion reconstructs targets while preserving known context.

See [model.py](dss_imputer/model.py), [diffusion.py](dss_imputer/diffusion.py),
[masks.py](dss_imputer/masks.py), and [schema.py](dss_imputer/schema.py).
LOCO means hiding one entire nonconstant column at a time. MCAR training uses
random cell masks with constraints ensuring usable targets and context; its
configured probability is not an exact fraction of every row.

## Local Smoke Run

This example explicitly selects Hybrid, MCAR 15%, logarithmic IOPS/latency,
200 diffusion steps, and learning rate `5e-5`. These are experiment settings:
the direct training CLI defaults to MCAR 30%, no logarithm, and `2e-4`.

```bash
python scripts/train_dss_imputer.py \
  --dataset-config configs/dss_imputer/datasets.json --dataset hdd_sequential \
  --output-dir trained_models/dss_imputer/hdd_smoke \
  --architecture hybrid --d-model 64 --layers 1 --heads 4 --mlp-width 128 \
  --epochs 2 --batch-size 512 --learning-rate 5e-5 --patience 0 \
  --diffusion-steps 200 --min-mask-rate 0.15 --max-mask-rate 0.15 \
  --log-continuous-cols iops lat

python scripts/eval_dss_imputer.py \
  --data-csv dataset/pools/test_hdd_sequential.csv --split-label test \
  --checkpoint trained_models/dss_imputer/hdd_smoke/last.pt \
  --output-dir reports/dss_imputer/hdd_smoke/test_last \
  --patterns loco --loco-cols iops lat --samples 5 --limit 512 \
  --save-samples --generative-metrics

python scripts/plot_dss_imputer_loco_quality.py \
  --results-dir reports/dss_imputer/hdd_smoke/test_last \
  --prediction-source trial_000 --pair-plot-kind scatter \
  --plot-sample-size 0 --metric-sample-size 0
```

The smoke example still trains on the full input CSV. Omit `--limit` for full
evaluation and omit `--loco-cols` to evaluate every modeled nonconstant column.
For train evaluation, change `--data-csv`, `--split-label`, and the output directory
to the training split; use `--training-subset all`. This creates new stochastic
samples, not just plots. Plotting an existing results directory does not resample.

Checkpoints are `best.pt`, `last.pt`, and `epoch_XXXX.pt` every 100 epochs by
default. `--patience 0` disables early stopping. Continuation via
`--resume-checkpoint` restores weights and preprocessing but uses a **fresh
optimizer/scheduler**; it is not an exact interrupted-training resume.

## SLURM: Train, Evaluate, Plot

On the cluster, prepare the environment and log directory **before** submitting:

```bash
cd /home/kkatsuba/work_flow/dss_synth_generation
mkdir -p slurm/logs
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CONDA_ENV=tabdiff
export PLOT_CONDA_ENV=tabdiff
export CLEARML_PROJECT=""
export DATASET_CONFIG=configs/dss_imputer/datasets.json
# For cleaned data, use dataset/cleaned/datasets.json instead.

DATASETS="hdd_sequential ssd_sequential ssd_random cache" \
RUN_TAG=hybrid_mcar15_log_t200_e500_s42 \
PHASE=pipeline ARCHITECTURE=hybrid \
EPOCHS=500 PATIENCE=0 LEARNING_RATE=5e-5 DIFFUSION_STEPS=200 \
MIN_MASK_RATE=0.15 MAX_MASK_RATE=0.15 \
CHECKPOINTS="best last" \
TRAIN_TIME=05:00:00 EVAL_TIME=05:00:00 PLOT_TIME=00:15:00 \
bash slurm/submit_dss_imputer_all.sh -A proj_1808
```

Replace the account with one available to you. Jobs use the `rocky` partition;
training/evaluation request one GPU and four CPUs, plotting runs on CPU. Memory
allocation is left to cluster defaults. Use fresh run tags when changing data or
configuration to avoid mixing artifacts from different experiments.

This submits training followed by dependent test LOCO evaluation and plots for
each checkpoint. Runs are named `<dataset>_<RUN_TAG>`. Defaults in this launcher
include batch size 512, five evaluation trials, all nonconstant LOCO targets,
saved trial CSVs, generative metrics, and full-row plots. Metric computation is
subsampled separately. To train Cache for 1,000 epochs, submit it separately with
`DATASETS=cache EPOCHS=1000` and a distinct run tag; exclude it from the first call.
`DRY_RUN=1` previews submissions, but still requires the relevant input files.

The launcher records job IDs in `slurm/logs/`. For an existing run, `PHASE=reports`
evaluates/plots train and plots the existing test samples. Consult the launcher
before using other phases; running sampling again is different from redrawing
saved trials. Watch jobs with `squeue -u "$USER"`, inspect completed jobs with
`sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed`, and follow logs with
`tail -F slurm/logs/<job-name>-<job-id>.out`.

## Baselines and Comparison

`scripts/train_external_generators.py` orchestrates upstream clone, data
preparation, training, and native sampling. The matching SLURM files are
`prepare_external_generators.sbatch`, `train_tabdiff.sbatch`,
`train_tabsyn.sbatch`, and `train_tabddpm.sbatch`. Use the upstream model's own
environment for training and sampling. See each script's `--help` and these guides:

- [Shared cleaned-data preparation and baseline training](docs/cloud_outlier_cleaning.md)
- [Unconditional sampling, no-CFG LOCO, metrics and plots](docs/external_generators_evaluation.md)
- [Single-trial LOCO comparison](docs/loco_single_trial_comparison.md)
- [GT / DSS / TabDiff / TabSyn distribution and scatter exports](docs/loco_four_panel_comparison.md)

External models first learn unconditional generation. No-CFG LOCO uses native
inpainting for TabDiff and latent RePaint-style inpainting for TabSyn. Their
sampling can forward-noise/clamp known context internally before restoring it;
this differs from DSS Imputer's permanently clean context. Neither procedure is
claimed to be exact sampling from the true conditional distribution.

## Reading Metrics

- Prioritize **target-only generative scores** for LOCO. Full-table scores include
  unchanged context and can hide poor reconstruction of the missing column.
- SDMetrics measures column shapes and pair trends. The reported `c2st` is
  **LogisticDetection quality (higher is better)**, not raw classifier accuracy.
- Alpha precision, beta recall, and authenticity use the project's naive embedding
  implementation; they are not automatically identical to every paper's protocol.
- DCR is descriptive for conditional imputation, not a privacy guarantee: context
  is copied from real rows. Train LOCO DCR is omitted when references overlap.
- Compute metrics separately for each stochastic trial, then report mean +/- std
  and the valid-trial count. Computing metrics on an averaged prediction answers
  a different question and does not measure the generated distribution.
- Numerical R2/MSE/RMSE/MAE and categorical accuracy/F1 are complementary predictive
  scores. KS and TV in comparison reports assess marginal distribution differences.
- Keep schema, row selection, masks, seeds, and metric versions aligned across
  models. Current protocol: `explicit_schema_masked_targets_c2st_union_v3`.
  Inspect metric error files; a missing score is not a zero or a successful result.

Evaluation saves `run_info.json`, metric tables, and per-scenario outputs such as
`loco_iops/trial_000.csv`. Saving every trial requires `--save-samples`. Plotters
use `trial_000` by default. Plot limits and metric sample limits are independent
of the number of generated rows; fixed IOPS/latency display windows deliberately
hide points outside those windows.

`analyze_clean_loco_trials.py` and `summarize_clean_loco_trials.py` reproduce a
particular downloaded experiment layout; their defaults are not a generic data
discovery mechanism. Use the general evaluation/comparison tools for new runs.

## Tests and Reproducibility

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Tests exercise preprocessing, masking, diffusion, context preservation, trial
metrics, plotting, and SLURM command construction using small fixtures and mocked
upstream integrations. They do not replace an actual GPU training run in each
baseline environment. On macOS environments with OpenMP fork/shared-memory
issues, prefix tests with `KMP_INIT_AT_FORK=FALSE OMP_NUM_THREADS=1`.

Archive run configs, data hashes, training history, metric protocol, logs, and the
Git commit alongside results. Seeds alone do not guarantee bitwise-identical
results across hardware and dependency versions. Do not commit ClearML/API keys
or SSH keys. Cloned baseline projects retain their upstream licenses.
