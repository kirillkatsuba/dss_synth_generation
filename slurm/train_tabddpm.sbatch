#!/bin/bash
#SBATCH --job-name=dss_tabddpm_train
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=05:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-tddpm}"
CONDA_SH="${CONDA_SH:-}"
TARGETS="${TARGETS:-iops}"
DEVICE="${DEVICE:-cuda:0}"
TABDDPM_STEPS="${TABDDPM_STEPS:-10000}"


cd "${PROJECT_DIR}"
mkdir -p slurm/logs

if [[ -n "${CONDA_ENV}" ]]; then
  if [[ -z "${CONDA_SH}" ]]; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  fi
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

"${PYTHON_BIN}" scripts/train_external_generators.py \
  --phase train \
  --models tabddpm \
  --targets ${TARGETS} \
  --python "${PYTHON_BIN}" \
  --device "${DEVICE}" \
  --tabddpm-steps "${TABDDPM_STEPS}"
