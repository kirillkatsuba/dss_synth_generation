# Synthetic Tabular Data Generation for Data Storage Systems

## CTGAN
Training procedure:
```[python]
python3 ctgan_generator.py {dataset_name}.csv \
  --output-model /home/kkatsuba/work_flow/dss_generation/trained_models/ctgan/ctgan_model_{dataset_name}.pkl \
  --gpu \
  --verbose \
  --output-data /home/kkatsuba/work_flow/dss_generation/synth_data/ctgan/ctgan_synth_{dataset_name}.csv
```

Sampling:
```[python]
python3 ctgan_generator.py {dataset_name}.csv \
  --sample \
  --load-model /home/kkatsuba/work_flow/dss_generation/trained_models/ctgan/ctgan_model_{dataset_name}.pkl \
  --output-data /home/kkatsuba/work_flow/dss_generation/synth_data/ctgan/ctgan_synth_{dataset_name}.csv \
  --random-seed 42
```
# dss_synth_generation
Synthetic Tabular Data Generation for Data Storage Systems

## External diffusion generators

CTGAN is implemented locally in `scripts/ctgan_generator.py`. The remaining
generators are wrapped by `scripts/train_external_generators.py`:

- TabDDPM: `external/tab-ddpm`
- TabDiff: `external/tabdiff`
- TabSyn: `external/tabsyn`

The script prepares upstream data/configs and then runs training and sampling.
The external repositories require one target column in metadata, but the sampled
CSV still contains all original columns. By default the wrapper uses `iops` as
that metadata target and produces one synthetic dataset per model.

```bash
# Clone missing repositories and prepare data/configs.
python3 scripts/train_external_generators.py --phase clone prepare --models all

# See the exact train/sample commands without running long jobs.
python3 scripts/train_external_generators.py --phase train sample --models all --dry-run

# Train all three generators.
python3 scripts/train_external_generators.py --phase train --models all

# Sample trained models and copy final CSV files into synth_data/.
python3 scripts/train_external_generators.py --phase sample --models all
```

Each upstream project has its own dependency set. Run the wrapper with
`--python /path/to/env/bin/python` when using a dedicated environment.

If a second metadata target is needed later, pass it explicitly:

```bash
python3 scripts/train_external_generators.py --phase prepare train sample --models all --targets iops lat
```
