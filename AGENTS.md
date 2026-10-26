# Repository Instructions

## SLURM

Do not specify memory resources in SLURM scripts, submission commands, or
examples. Omit `--mem`, `--mem-per-cpu`, `--mem-per-gpu`, and equivalent
`SBATCH_MEM_*` settings. Leave memory allocation to the cluster defaults.
