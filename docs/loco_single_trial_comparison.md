# Single-trial LOCO comparison

`scripts/compare_loco_single_trial.py` compares saved predictions; it does not
train models or run inference. The input reports remain unchanged.

```bash
python3 scripts/compare_loco_single_trial.py \
  --reports-root /Users/kirill/edu/reports \
  --output-dir reports/loco_single_trial_comparison
```

## Selection

- Every nonconstant modeled column is a separate LOCO target.
- Train and test are evaluated separately for all four datasets.
- Hybrid uses `last`, including the older HDD run `hdd_hybrid_mcar15_log_s42`.
- TabSyn and TabDiff use the checkpoints recorded in their source reports.
- Only `trial_000.csv` is used, without averaging or selecting the best trial.
- Cache t400/epoch1000 is an additional test-only comparison, separate from the
  main t200 cache experiment.

Known context is first checked against each model's own saved ground truth.
Rows are then matched by all modeled ground-truth columns at float32 precision,
with occurrence indices for duplicates. This accommodates external preprocessing
rounding without rounding any predictions. Only known context is harmonized to
the shared GT precision after the integrity check. Matching and precision
differences are recorded in `alignment_audit.csv`.

HDD Hybrid's saved training evaluation contains only its internal fit subset.
Its comparison uses the common 66,218 rows, not all 73,596 external train rows.
The models therefore do not have identical historical training sets.

## Metrics

Generative metrics use the same deterministic subset, up to 2,000 matched rows,
for all three methods. MSE and R2 use all matched rows, in original units, not
logarithmic space or the scaled plot units. Numeric-valued categorical targets
also receive MSE/R2 without rounding. Text-valued categories do not.

The common evaluation schema treats `iops`, `lat`, `n_jobs`, `iodepth` as numerical
and all remaining modeled columns as categorical. This is the Hybrid schema,
applied consistently to every method. It does not alter model predictions.

- Target shape and target-context pair trends come from SDMetrics QualityReport.
  Their mean is the target overall quality. Known-known pairs are not included
  in these target-specific scores.
- Joint C2ST is SDMetrics LogisticDetection quality. Higher is better; this is
  not classifier accuracy. Its linear discriminator can miss nonlinear errors.
- Joint alpha-precision and beta-recall use the existing naive geometric
  estimator, with numerical min-max scaling and categorical one-hot encoding.
- Target-only C2ST, alpha and beta are also preserved in the detailed CSVs.
  A correct marginal is insufficient to establish correct conditional sampling.
- Joint metrics include the copied context and must be interpreted alongside
  target-context scores and plots. Discrete target-only alpha/beta can be
  degenerate because of distance ties.
- DCR is deliberately omitted: copied conditional context and overlapping train
  references complicate privacy interpretations.

One trial provides no between-trial standard deviation or confidence interval.
Per-target scores should be examined before attempting a model ranking.

## Figures

Every 1D figure has three columns: Hybrid, TabSyn, TabDiff, each compared to the
same GT. Numeric-valued targets use filled histograms with shared bins. Text
categories use probability bars. IOPS/latency also have log-scale versions.

Every pair figure has two rows and three columns: GT above and prediction below.
Each LOCO target is compared to every other modeled column, used as context.
Scatter panels preserve physical coordinates for numeric-valued columns.
Additional type-aware panels show boxplots for categorical context/numerical
target, and conditional probability heatmaps for categorical targets.
Heatmap categories not present in GT are grouped under `Outside GT` for display
only. If a column has more than 12 GT categories, rarer levels are grouped under
`Other GT`, distinct from `Outside GT`. The metric calculation does not group
any categories. Estimator warnings are preserved in `metric_errors.csv`.
`Outside GT` means absent from the evaluated reference split, not necessarily
physically invalid or unseen during training.

All matched rows contribute to the plots. HDD scatter windows retain IOPS/1,000
in [0,14] and latency/1,000,000 in [0,100], with out-of-window fractions annotated.
Full-range IOPS/latency pair figures are supplied too. No clipping is applied to
the metrics.

## Outputs

Open `reports/loco_single_trial_comparison/index.html`. It links the separate
dataset/split galleries, combined metric tables and audit CSVs. Each target's
folder contains its combined figures and per-model cached metric JSON.

`--phase metrics` computes only metrics; `--phase plots` requires matching caches.
The default `--phase all` computes missing metrics and builds all figures.
`--no-include-cache400` excludes the extra cache experiment.
`--skip-existing-plots` resumes an interrupted plot run without validating the
existing PNGs; use it only when the input data and plot settings are unchanged.
