#!/bin/bash
#SBATCH --job-name=dss_prepare_synth
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --nodes=1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV="${CONDA_ENV:-pytorch_new}"
MODELS="${MODELS:-all}"
TARGETS="${TARGETS:-iops}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

"${PYTHON_BIN}" scripts/train_external_generators.py \
  --phase clone prepare \
  --models ${MODELS} \
  --targets ${TARGETS}
