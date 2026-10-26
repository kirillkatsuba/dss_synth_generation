# Four-panel LOCO export

## Minimal Scatter Report

```bash
python3 scripts/export_loco_comparison.py --minimal
```

This packages the already verified four-panel report into
`reports/loco_scatter_report` without inference or metric recomputation:

```text
<dataset>/loco_<target>/
  train_distribution.png
  test_distribution.png
  train_scatter_vs_<context>.png
  test_scatter_vs_<context>.png
  train_metrics.png
  test_metrics.png
  train_metrics.csv
  test_metrics.csv
  index.html
```

Only the four main dataset runs are included. No boxplots, heatmaps, duplicate
log/full-range views, extra cache run, MSE/R2, audit files or per-metric matrix
directories are copied. The tables contain generative quality scores and TV/KS
only, with actual underlined winners in both PNG and HTML. Their numerical
definitions, ties and convergence-warning rules remain unchanged. CSV is plain
data and cannot carry underlining. Root `index.html` links the LOCO folders.

## Full Export

Run from the project root:

```bash
python3 scripts/export_loco_comparison.py
```

The source reports remain in `/Users/kirill/edu/reports`. The previous comparison
remains in `reports/loco_single_trial_comparison`. The new report is written to
`reports/loco_single_trial_four_panel/index.html`.

This exporter reuses row alignment, context checks and verified cached generative
metrics from `scripts/compare_loco_single_trial.py`. It never trains or samples.
It uses only saved `trial_000.csv`, Hybrid `last`, and the existing source
checkpoints for TabSyn/TabDiff. The source checkpoint caveats of the previous
comparison remain applicable.

## Organization

```text
<dataset>/<train|test>/main/
  index.html
  metrics.csv
  metrics_table.png
  tables/
    index.html
    tv.csv, tv.png
    ks.csv, ks.png
    <other metric>.csv, <other metric>.png
  loco_<target>/
    index.html
    metrics.csv
    metrics_table.png
    marginal_protocol.json
    distribution.png
    distribution_log.png        # IOPS and latency only
    pair_<context>.png
    conditional_<context>.png   # pairs involving semantic categories
```

The extra cache t400 epoch1000 test comparison uses `cache/test/cache_t400`.
Main train/test reports for cache still use the t200 run, without mixing runs.

One-dimensional panels use four columns, ordered GT, Ours, TabDiff, TabSyn.
Pairs use a 2x2 grid in the same row-major order. GT occurs once and all panels
share axis limits. Every other modeled column is used as context in a separate
pair figure. Numeric-valued columns retain scatter views; additional semantic
categorical views use conditional boxplots or conditional frequency heatmaps.
Display-only rare-category groups do not affect any metric.

## TV and KS

The new marginal metrics use **all aligned rows**, without averaging trials,
clipping, log transforms, prediction rounding or sampling.

TV is a distance, so **lower is better**:

```text
TV(P,Q) = 0.5 * sum_k |P(k) - Q(k)|.
```

For semantic categorical columns it is the exact empirical PMF distance over the
union of observed and generated levels. This includes numeric-valued categories.
The same estimator is used for integer numerical columns with at most 128 unique
GT values and for a constant reference. Noninteger predictions are not rounded
to valid integer values: off-grid outputs are retained as separate levels.

For continuous numerical columns, exact empirical TV would be dominated by
nonmatching sample locations. Instead, the report explicitly uses a **binned TV
approximation**. All models use the same GT 50-quantile edges, with repeated edges
removed and additional underflow/overflow bins. Predictions beyond the GT range
therefore contribute their full mass, rather than being discarded. The upper GT
edge is moved to its next representable float so the GT maximum remains in the
interior, not the overflow bin. Exact edges are recorded in each LOCO folder.
This binning is distinct from the equal-width display histograms. No KDE is used
for the metric. Interpretation depends on the documented binning resolution.

KS is the two-sample statistic on the original numeric values:

```text
KS(P,Q) = sup_x |F_P(x) - F_Q(x)|.
```

It is computed for continuous, integer and numeric-valued categorical columns.
For unordered text categories, KS is `n/a`, never computed on arbitrary codes.
Only the distance is reported, not a p-value or significance claim.

Other generative metrics retain their deterministic 2000-row cap, unless changed
with `--metric-sample-size`. MSE/R2 and plots continue to use all aligned rows.
HDD training comparison remains limited to the shared 66,218-row subset.

## Best-Value Highlighting

Values are compared **within a dataset, split, run variant and LOCO target**.
TV/KS/MSE are minimized; the other displayed quality scores and R2 are maximized.
All ties are bold, using unrounded scores with rtol=1e-9 and atol=1e-12.
Text-category KS/MSE/R2 and other nonfinite scores are not ranked. C2ST values
associated with a convergence warning are preserved with an asterisk, but are
excluded from winner selection. A highlight does not establish statistical
significance from this single trial.

Bold green winners appear in HTML and PNG tables. CSV cannot carry rich text;
`metrics_best_flags.csv` records boolean winner flags alongside the raw CSVs.
Each dataset/split also contains one target-by-model table per metric, under
`tables/`, with the same ranking rules.

`--skip-existing-plots` resumes image generation without validating existing PNGs.
Use it only with unchanged input data and plot settings. Tables and marginal
scores are rebuilt regardless. Cache signatures validate source paths, sizes,
timestamps, row counts, metric cap and the shared generative implementation.
