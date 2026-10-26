# TabSyn / TabDiff: unconditional generation and no-CFG LOCO

This workflow evaluates the already trained models for all four DSS datasets.
It never trains a model or changes the prepared datasets or checkpoint schema.
The default training run tag matches `uncond_all_20261015`.

## Transfer

Run on the local machine:

```bash
cd /Users/kirill/edu/lambda/dss_synth_generation
scp scripts/eval_external_generators.py scripts/tabdiff_impute.py scripts/tabsyn_impute.py \
  scripts/plot_dss_imputer_loco_quality.py scripts/plot_dss_imputer_evaluation.py \
  charisma-hse:/home/kkatsuba/work_flow/dss_synth_generation/scripts/
scp dss_imputer/generative_metrics.py \
  charisma-hse:/home/kkatsuba/work_flow/dss_synth_generation/dss_imputer/
scp slurm/submit_external_evaluation.sh slurm/sample_external_generators.sbatch slurm/report_external_generators.sbatch \
  charisma-hse:/home/kkatsuba/work_flow/dss_synth_generation/slurm/
```

Existing dependencies on the cluster must also be current:
`scripts/plot_dss_imputer_evaluation.py`, `scripts/plot_dss_imputer_loco_quality.py`,
`scripts/eval_dss_imputer.py`, `scripts/recompute_dss_imputer_metrics.py`,
and the `dss_imputer` package. These are the same utilities used by DSS imputer
evaluation. Plotting / metrics require SDMetrics, pandas, NumPy, scikit-learn,
Matplotlib, Seaborn and PyTorch in the reporting environment.

## Stage 1: native unconditional samples, quality against train

```bash
ssh charisma-hse
cd /home/kkatsuba/work_flow/dss_synth_generation
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export RUN_TAG=uncond_all_20261015
export METRICS_ENV=pytorch_new

PHASE=unconditional TRIALS=5 \
  bash slurm/submit_external_evaluation.sh -A proj_1808
```

Default datasets: `hdd_sequential ssd_sequential ssd_random cache`.
Default models: `tabsyn tabdiff`.
For a smoke run, prefix the command with `DATASETS=cache MODELS=tabdiff`.
`DRY_RUN=1` prints the jobs without submitting them; checkpoints/data are still
checked before the dry run.

This stage submits 24 jobs: sample -> metrics -> plots for each of eight models.
Each of five independent synthetic tables has the same row count as the
prepared training table. No real row or context is passed to the denoiser.
TabDiff uses `sample_all()` and TabSyn uses its `diffusion_utils.sample()`;
the upstream TabDiff MLE evaluation is not invoked. Native TabDiff `sample_all`
uses its default NaN filtering (`keep_nan_samples=False`). TabSyn is batched
to bound GPU usage without changing its EDM sampling steps.

Inspect the unconditional metrics and plots before starting the LOCO stage.
No automated threshold declares that generated data is plausible.

## Stage 2: LOCO on complete train and test

```bash
PHASE=loco TRIALS=5 \
  bash slurm/submit_external_evaluation.sh -A proj_1808
```

This stage submits 48 jobs: sample -> metrics -> plots for each model and split.
One sampling job runs every nonconstant LOCO target sequentially. There is no
row limit: both sampling and default plots use the entire prepared split.
The inference environment is `tabsyn` or `tabdiff`; CPU metrics and plots use
`METRICS_ENV`, defaulting to `pytorch_new`. No ClearML or W&B task is initialized.

For TabDiff, `y_only_model=None` and both guidance weights are zero.
For TabSyn, inference uses latent RePaint-style inpainting without CFG.
The default condition is `x_t`, with one resampling round. Known context is
forward-noised/clamped inside the baseline sampling process and restored exactly
in the final table. This is not the same conditioning architecture as DSS imputer
with permanently clean context; it is not a claim of exact posterior sampling.

All known cells are checked for changes. Each target directory has
`trial_000.csv` ... `trial_004.csv`, `target_mask.csv`, and the legacy point
estimate. Generative metrics and plots never use that point estimate.

`PHASE=all` submits both stages together. LOCO waits for successful unconditional
metric computation, not for a human review or a quality threshold.

## Outputs

```text
reports/external_generators/uncond_all_20261015/
  tabsyn|tabdiff/
    hdd_sequential|ssd_sequential|ssd_random|cache/
      unconditional/
      loco_train/
      loco_test/
```

Every result root contains `evaluation_truth.csv`, `run_info.json`, stochastic
trial subdirectories, `generative_metrics_by_trial.csv`,
`generative_metrics_summary.csv`, and `generative_metric_errors.csv`.
LOCO results also contain `context_integrity.csv`.

`quality_plots` contains one-dimensional comparisons for every target. LOCO
plots additionally compare each generated target with every known modeled
column using the existing DSS imputer scatter plotter, including categorical
axis labels. Pairwise SDMetrics scores are saved in `pairwise_metrics.csv`.
Unconditional plots include all-column histograms and real/generated
IOPS-versus-latency scatter plots. Plot source: `trial_000.csv`.

## Metric protocol

- Quality: SDMetrics column shapes and column pair trends. Unconditional quality
  is measured against real train, not against paired row-wise predictions.
- C2ST: SDMetrics LogisticDetection quality, larger is better. A shared one-hot
  support preserves categories generated outside the real comparison sample.
  This fixes the previous unknown-category crash. The protocol version is
  `explicit_schema_masked_targets_c2st_union_v3`; old C2ST values must be
  recomputed with the same protocol before comparison.
- Alpha precision, beta recall and authenticity: the same naive embedding
  implementation as DSS imputer.
- LOCO: target-only and full-table metrics, separately per stochastic trial,
  then mean and standard deviation across trials. Prefer target metrics because
  full-table scores include unchanged context.
- Unconditional DCR uses distinct prepared train and held-out test references.
  DCR for test LOCO is descriptive, not a privacy guarantee, because real
  context is copied. Train LOCO DCR is omitted rather than using overlapping
  real/train references.
- Metric subsampling defaults to 5,000 rows and DCR to 1,000 rows. The limits
  and successful trial counts are recorded. This subsampling does not restrict
  generation or plots. Plot and pairwise-score limits default to zero (all rows).
- Model column types come from each trained external model's metadata. They
  are not silently replaced with the DSS imputer schema. Numeric baseline
  features such as `read_fraction` / `block_size` may differ in type from DSS.

## Resources and reruns

Sampling defaults to 5 hours, one GPU, one node and four CPUs in `rocky`.
Metrics default to 5 hours on CPU; plots to 15 minutes on CPU. Override with
`SAMPLE_TIME`, `METRIC_TIME`, and `PLOT_TIME`. No explicit allocation size for
RAM is requested.

`METRIC_SAMPLE_SIZE`, `DCR_SAMPLE_SIZE`, `PLOT_SAMPLE_SIZE`,
`PAIR_METRIC_SAMPLE_SIZE`, `BATCH_SIZE`, `STEPS` (TabSyn), `TRIALS`, and
`RESAMPLE_ROUNDS` can be set in the environment.

Saved trials are protected against accidental overwrite. To recalculate
metrics and plots from existing samples without generating anything:

```bash
PHASE=reports bash slurm/submit_external_evaluation.sh -A proj_1808
```

Submitted job IDs and dependencies are recorded in a TSV manifest under
`slurm/logs/external_eval_*.tsv`. A failed submission stops the loop and keeps
the IDs already submitted; it does not cancel those jobs or retry blindly.
