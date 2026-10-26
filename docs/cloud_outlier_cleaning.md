# Cloud-wise Train Cleaning

```bash
python3 scripts/clean_dataset_outliers.py
```

Reads the four configured train CSVs and writes independent copies plus visual
comparisons to `reports/outlier_cleaning`. It does not update dataset configuration,
train models, change input files or fit/filter test data (the test file is only
hashed for provenance). To inspect a single
dataset, pass `--datasets hdd_sequential`. Existing output directories require
an explicit `--overwrite` or a different `--output-dir`.

Each `id + io_type` group is fitted independently with
`EllipticEnvelope(contamination=0.05)`, using only
`[log(iops + epsilon), log(lat + epsilon)]`. Epsilon defaults to `1e-6` and
the reproducible seed defaults to 42, with stable group-specific derived seeds.
The detector adapts the function in HSE-LAMBDA/digital-twin commit
`884cd9a0c334dfb96ab4291a4c734d3ae848ccdf`, `scripts/calculate_summary_stats.py`.

The resulting mask selects complete original rows. The cleaned CSV retains all
original columns and their original values and order, without log-transforming
or clipping stored IOPS/latency. Non-target context must be constant inside each
cloud; mixed contexts fail explicitly instead of being combined.

The contamination setting controls the approximate fraction rejected within each
fitted cloud. It is not an estimated anomaly prevalence or a guarantee that all
remaining rows are physically valid. Small clouds (<20 rows), rank-deficient
data and unstable/failed covariance estimates are retained and audited.

Zeros are allowed by default (`--zero-policy keep`), enter the detector through
`log(x + epsilon)`, and are counted separately. They are not automatically removed.
`--zero-policy drop` instead explicitly excludes rows where either measurement
equals zero, before fitting. Nonfinite and negative measurements fail rather
than being silently removed. These policies must not be conflated with the
elliptic 5% trimming itself.

Each dataset folder contains:

- `train_clean.csv`: retained original rows, ready for later training.
- `removed_rows.csv`: removed original rows with zero-based `source_row`.
- `row_decisions.csv`: keep mask, reason, zero flag, squared Mahalanobis distance.
- `cloud_summary.csv`: per-cloud counts, skip statuses and estimator warnings.
- `run_info.json`: source path/hash, parameters and total counts.
- `before_after.png`: full-range linear scatter, original left / cleaned right.
- `before_after_log.png`: corresponding scatter in the detector coordinates.
- `before_after_window.png`: HDD-only established IOPS/1000 [0,14] and
  latency/1000000 [0,100] window, with counts outside the window annotated.
- `cloud_examples.png`: four individual clouds, prioritizing zero measurements
  and removed-point counts. Disable with `--cloud-examples 0`.

In scatter comparisons, removed rows are marked red on the original left panel;
the retained rows are blue on the right. Both panels share limits determined
from the original data, not independent autoscaling. All rows contribute to the
full-range figures. Plot scaling by 1,000 and 1,000,000 is display-only and does
not assert a physical latency unit conversion.

The root `index.html` is the visual entry point; `summary.csv` contains totals.

## Shared preparation for training

### Reuse the already inspected CSVs

`configs/dss_imputer/datasets_clean.json` points to the existing
`reports/outlier_cleaning/<dataset>/train_clean.csv` for all four datasets and
the original unchanged test CSVs. To reuse exactly the rows inspected locally,
transfer those four CSVs, this config, the updated Python entry points and
`scripts/training_data.py` to the cluster, preserving relative paths. Export:

```bash
export DATASET_CONFIG=configs/dss_imputer/datasets_clean.json
```

Do not run `prepare_training_data.sbatch` for this route: no further outlier
filtering is needed. External generators still need their model-specific
`prepare_external_generators.sbatch` stage on these files; the multi-dataset
launcher submits it before the dependent training jobs. Use a fresh
`DATASET_SUFFIX`, such as `_clean_20261022`, and fresh DSS output directories to
avoid replacing earlier models. Cleaned CSVs and configs must be identical
throughout preparation, training and later evaluation.

The updated training jobs use partition `rocky`, one GPU, one node and four CPUs.
Their existing model-specific time limits are unchanged. The DSS job resolves
the config before printing its input path, so `Train CSV` reports the actual
cleaned file rather than the overridden raw-data default.

### Generate a new cleaning experiment

Cleaning is a separate, explicit preparation stage, before model-specific
normalization, category encoding, constant-column removal and training. Run once
and reuse exactly the same retained rows for DSS Imputer, TabDiff, TabSyn and
TabDDPM. Never fit a detector to test data or remove test rows using their target
values. The original test distribution remains the evaluation benchmark.

```bash
python3 scripts/clean_dataset_outliers.py \
  --output-dir dataset/cleaned \
  --prepared-config dataset/cleaned/datasets.json
```

This prepares all four datasets by default. For each dataset it writes
`dataset/cleaned/<dataset>/train_clean.csv` plus all masks, cloud audits and
before/after plots listed above. `dataset/cleaned/datasets.json` maps the dataset
keys to these cleaned training files and the **original unchanged test files**.
Paths inside the repository are repository-relative, so the configuration can be
regenerated or transferred without hardcoded local-machine paths. Sources outside
the repository have absolute paths. Do not overwrite the original config.

To change the fraction, pass `--contamination`; default `0.05` matches the local
experiment. To regenerate an existing output use `--overwrite` deliberately, or
prefer a new output directory/config for a different cleaning experiment.
Regeneration with only `--datasets cache`, for example, produces a config only
for Cache; it does not merge with an old four-dataset configuration.

### SLURM preparation

```bash
cd /home/kkatsuba/work_flow/dss_synth_generation
mkdir -p slurm/logs
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV=pytorch_new sbatch -A proj_1808 --export=ALL slurm/prepare_training_data.sbatch
```

This CPU job needs numpy, pandas, scikit-learn and matplotlib. It does not start
training. Wait for successful completion before using the prepared config.
`DATASETS`, `CLEAN_DATA_DIR`, `SOURCE_CONFIG`, `OUTLIER_CONTAMINATION`,
`OUTLIER_SEED`, `OUTLIER_EPSILON` and `OVERWRITE=1` can be supplied as environment
variables. An individual failed/degenerate cloud is retained and recorded, as
before; a source validation error fails the entire preparation job.

### DSS Imputer

```bash
export DATASET_CONFIG=dataset/cleaned/datasets.json
export DATASET=hdd_sequential
CONDA_ENV=tabdiff sbatch -A proj_1808 --export=ALL slurm/train_dss_imputer.sbatch
```

All other architecture/training options work as before. For the existing complete
train/eval/plots launcher, after preparation:

```bash
DATASET_CONFIG=dataset/cleaned/datasets.json \
DATASETS="hdd_sequential ssd_sequential ssd_random cache" \
PHASE=pipeline bash slurm/submit_dss_imputer_all.sh -A proj_1808
```

The default run names gain `_clean`; explicitly supplied `RUN_TAG`, `OUTPUT_DIR`
and `CHECKPOINT` still take precedence, so use fresh names for new experiments.
`DATASET_CONFIG` also selects the cleaned train for `SPLIT=train` evaluation and
the unchanged test for `SPLIT=test` or `validation`. Keep it exported for later
eval jobs. This does not change the existing checkpoint-selection protocol: if
test is used for selecting `best`, it is not an independent final holdout.

The direct Python interface is:

```bash
python scripts/train_dss_imputer.py \
  --dataset-config dataset/cleaned/datasets.json --dataset hdd_sequential \
  --output-dir trained_models/dss_imputer/hdd_clean
```

The config overrides `--train-csv` and `--validation-csv` when supplied. Training
uses the entire cleaned train, except any explicit `--max-rows` subsampling;
preprocessing is fitted only on that train. Start from scratch for this comparison,
not an old checkpoint with raw-data normalization/vocabulary.

### External generators

```bash
export DATASET_CONFIG=dataset/cleaned/datasets.json
export DATASET=hdd_sequential
PREPARE_JOB=$(CONDA_ENV=pytorch_new sbatch --parsable -A proj_1808 --export=ALL slurm/prepare_external_generators.sbatch)
CONDA_ENV=tabdiff sbatch -A proj_1808 --dependency="afterok:${PREPARE_JOB}" --export=ALL slurm/train_tabdiff.sbatch
CONDA_ENV=tabsyn sbatch -A proj_1808 --dependency="afterok:${PREPARE_JOB}" --export=ALL slurm/train_tabsyn.sbatch
CONDA_ENV=tddpm sbatch -A proj_1808 --dependency="afterok:${PREPARE_JOB}" --export=ALL slurm/train_tabddpm.sbatch
```

The preparation, training and individual sampling wrappers all accept
`DATASET_CONFIG` plus `DATASET`. Keep both identical across phases. Their default
upstream dataset prefix becomes `dss_<dataset>_clean`, avoiding the old raw-data
directories. `DATASET_PREFIX` still overrides this; choose a new name, not an old
trained dataset name. For all four datasets after the cleaning job succeeds:

```bash
DATASET_CONFIG=dataset/cleaned/datasets.json \
DATASETS="hdd_sequential ssd_sequential ssd_random cache" \
bash slurm/submit_remaining_datasets.sh -A proj_1808
```

This launcher passes the dataset key to every job and defaults to `_clean`
dataset prefixes and separate synthetic output paths. TabDDPM still makes its
existing 90/10 train/validation split **after cleaning**; TabSyn/TabDiff receive
the full cleaned train before their upstream preprocessing.

For direct Python commands use the same `--dataset-config ... --dataset ...`
arguments with `scripts/train_external_generators.py` for every phase. These
override `--real-train/--real-test`. For the newer evaluation pipeline use
`DATANAME=dss_<dataset>_clean_iops`; it reads the prepared upstream train/test, so
no further filtering occurs during sampling or scoring. To use the uncleaned
pipeline again, unset `DATASET_CONFIG` and choose the corresponding raw run names.

Transfer the updated `scripts/clean_dataset_outliers.py`, `scripts/training_data.py`,
`scripts/train_dss_imputer.py`, `scripts/train_external_generators.py`, the original
`configs/dss_imputer/datasets.json`, and the updated SLURM wrappers/launchers. The
new CPU job is `slurm/prepare_training_data.sbatch`. No upstream repository
patches or model architecture changes are needed.

## Restart DSS without early stopping; evaluate trained baselines

The DSS Python entry point and training SBATCH now default to `patience=0`.
This disables early stopping but still validates and saves `best.pt`, `last.pt`
and periodic snapshots. A positive explicit `PATIENCE` still enables stopping.
The DSS flag is not forwarded to TabSyn, TabDiff or TabDDPM; their upstream
training/stopping policies are independent and are not modified here.
Previously submitted jobs retain their submitted settings: editing a local
file or changing a shell export does not alter their running Python process.
The earlier explicit `PATIENCE=0` launch was already exempt from early stopping;
the remaining defaults of 15 were inconsistent with that launch.

### Transfer updated code (local machine)

```bash
cd /Users/kirill/edu/lambda/dss_synth_generation
rsync -avR scripts/train_dss_imputer.py \
  slurm/train_dss_imputer.sbatch slurm/submit_external_evaluation.sh \
  slurm/submit_sampling_and_reports.sh slurm/sample_tabddpm.sbatch \
  slurm/metric_eval.sbatch \
  charisma-hse:/home/kkatsuba/work_flow/dss_synth_generation/
```

The external workflow additionally requires the current
`sample_external_generators.sbatch`, `report_external_generators.sbatch`,
`eval_external_generators.py`, both external imputation scripts, the DSS plotting
and evaluation utilities and `dss_imputer/` described in
`external_generators_evaluation.md`. Transfer them too if the server has an
older copy. Neither cleaned CSVs nor upstream preparation need to be repeated.

### Server setup and fresh DSS training

```bash
ssh charisma-hse
cd /home/kkatsuba/work_flow/dss_synth_generation
mkdir -p slurm/logs
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" CLEARML_PROJECT=""
export DATASET_CONFIG=configs/dss_imputer/datasets_clean.json
export DATASETS="hdd_sequential ssd_sequential ssd_random cache"
unset PYTHON_BIN RESUME_CHECKPOINT ARTIFACT_SUFFIX TRAINING_LOCO_COL

# Only cancel the previous DSS jobs if replacing them; leave baselines alone.
scancel -u "$USER" --name=dss_imputer_train
# Wait until these jobs, including any CG jobs, have disappeared.
squeue -u "$USER" --name=dss_imputer_train

export ARCHITECTURE=hybrid D_MODEL=192 LAYERS=2 HEADS=8 MLP_WIDTH=512
export CONTINUOUS_COLS="iops lat n_jobs iodepth" LOG_CONTINUOUS_COLS="iops lat"
export EXCLUDE_COLS=id SEED=42 PATIENCE=0 CHECKPOINT_INTERVAL=100
export DIFFUSION_STEPS=200 BATCH_SIZE=512 LEARNING_RATE=5e-5 WEIGHT_DECAY=1e-4
export MIN_MASK_RATE=0.15 MAX_MASK_RATE=0.15
export MCAR_PROBABILITY=1 SINGLE_PROBABILITY=0 LOCO_PROBABILITY=0
export SEMANTIC_PROBABILITY=0 STRESS_PROBABILITY=0
export GENERATIVE_VALIDATION_INTERVAL=25 GENERATIVE_VALIDATION_ROWS=1024
export GENERATIVE_VALIDATION_RATE=0.15 GENERATIVE_VALIDATION_SAMPLES=1
export GENERATIVE_VALIDATION_LOCO_COLS="iops lat"
export TRAIN_TAG=clean_20261023_noes

for dataset in $DATASETS; do
  epochs=500
  [[ "$dataset" != cache ]] || epochs=1000
  DATASET="$dataset" EPOCHS="$epochs" CONDA_ENV=tabdiff \
  OUTPUT_DIR="trained_models/dss_imputer/${dataset}_hybrid_mcar15_log_t200_${TRAIN_TAG}_s42" \
    sbatch -A proj_1808 --time=06:00:00 --export=ALL slurm/train_dss_imputer.sbatch
done
```

Run the cancellation check before submission, not as a loop that could cancel
the new jobs. Use a fresh `TRAIN_TAG` on each new training attempt: no continuation
or reuse of old weights is intended. The six-hour allocation is a wall-clock
limit, not a promise that 1,000 epochs finish within it. A time limit can still
end training even when early stopping is off. Check `State`/`ExitCode` via `sacct`
and the final saved checkpoint's epoch before claiming completion.

### Trained TabSyn / TabDiff: sample, score and plot

Use the tag of the **completed external training**, not `TRAIN_TAG` above.
The following assumes the resubmission tag `clean_20261022_r2`; verify against
`external/tabsyn/data/dss_*clean*_iops` on the server. If a different tag was
actually submitted, replace it consistently in `RUN_TAG` and `DATASET_SUFFIX`.

```bash
export RUN_TAG=clean_20261022_r2
export TABDIFF_EXP_NAME_TEMPLATE='dss_tabdiff_dss_{dataset}_{run_tag}'
export REPORT_ROOT="reports/external_generators/${RUN_TAG}"
export METRICS_ENV=pytorch_new TRIALS=5 BATCH_SIZE=2048
export PLOT_SAMPLE_SIZE=0 PAIR_METRIC_SAMPLE_SIZE=0
PHASE=all MODELS="tabsyn tabdiff" \
  bash slurm/submit_external_evaluation.sh -A proj_1808
```

This submits native unconditional generation plus LOCO of every nonconstant
model column on full train and full test. CFG is disabled. Each sampling job
is followed by CPU metrics and then plots (15-minute plot limit).
`PLOT_SAMPLE_SIZE=0` uses all saved rows; metric subsampling still defaults to
5,000 rows. Outputs are under `$REPORT_ROOT/<model>/<dataset>/` in
`unconditional`, `loco_train` and `loco_test`, with `quality_plots` and CSV tables.
The launcher validates all prepared datasets and trained artifacts before
submission and refuses to overwrite saved trials. If samples already exist,
use `PHASE=reports` to recalculate metrics/plots without resampling.
Do not compare runs with different metric protocols without rescoring them.

The experiment-name template is optional: the default remains
`tabdiff_{dataset}_{run_tag}`, matching the older unconditional training run.
The explicit template above matches `submit_remaining_datasets.sh` and is used
both during checkpoint preflight and in the actual sampling job.

### Trained TabDDPM: unconditional sampling and the legacy report

```bash
DATASET_SUFFIX="_${RUN_TAG}" MODELS=tabddpm TARGETS=iops \
  bash slurm/submit_sampling_and_reports.sh -A proj_1808
```

This submits four sampling jobs in `tddpm`, followed by four metric/plot jobs in
`pytorch_new`. The shared dataset config resolves the cleaned real-train and the
unchanged real-test, and the suffix selects the already trained model folders.
Synthetic CSVs go to `synth_data/dss_<dataset>_<RUN_TAG>/`; reports go to
`reports/metric_eval_dss_<dataset>_<RUN_TAG>_tabddpm/`. This legacy report uses
`scripts/metric_eval.py`, not the newer no-CFG LOCO metric pipeline, and contains
generative metrics and additional downstream regression evaluation. These are
not automatically a protocol-matched three-way LOCO comparison. TabDDPM LOCO is
not implemented by this launcher; no such predictions are claimed.
