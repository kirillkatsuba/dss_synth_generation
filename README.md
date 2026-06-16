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

### SLURM jobs

Run preparation once on CPU:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,CONDA_ENV=pytorch_new slurm/prepare_external_generators.sbatch
```

Then submit one training job per model. Use the Python from the environment that
matches the upstream repository dependencies:

```bash
TABDDPM_JOB=$(sbatch --parsable --export=ALL,PROJECT_DIR=$PWD,CONDA_ENV=pytorch_new slurm/train_tabddpm.sbatch)
TABDIFF_JOB=$(sbatch --parsable --export=ALL,PROJECT_DIR=$PWD,CONDA_ENV=pytorch_new slurm/train_tabdiff.sbatch)
TABSYN_JOB=$(sbatch --parsable --export=ALL,PROJECT_DIR=$PWD,CONDA_ENV=pytorch_new slurm/train_tabsyn.sbatch)
```

After training, sample with dependencies so SLURM starts each sampling job only
if the corresponding training job finished successfully:

```bash
sbatch --dependency=afterok:$TABDDPM_JOB --export=ALL,PROJECT_DIR=$PWD,CONDA_ENV=pytorch_new slurm/sample_tabddpm.sbatch
sbatch --dependency=afterok:$TABDIFF_JOB --export=ALL,PROJECT_DIR=$PWD,CONDA_ENV=pytorch_new slurm/sample_tabdiff.sbatch
sbatch --dependency=afterok:$TABSYN_JOB --export=ALL,PROJECT_DIR=$PWD,CONDA_ENV=pytorch_new slurm/sample_tabsyn.sbatch
```

Edit `#SBATCH --partition`, `--gres`, memory, and walltime in `slurm/*.sbatch`
to match your cluster.

If `conda` is not available in non-interactive SLURM shells, pass the conda hook
explicitly:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,CONDA_ENV=pytorch_new,CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh slurm/train_tabsyn.sbatch
```
