#!/bin/bash
#SBATCH --job-name=dss_tabddpm_train
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/dss_synth_generation}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-}"
CONDA_SH="${CONDA_SH:-}"
TARGETS="${TARGETS:-iops}"
DEVICE="${DEVICE:-cuda:0}"
TABDDPM_STEPS="${TABDDPM_STEPS:-10000}"
CLEARML_PROJECT="${CLEARML_PROJECT:-}"
CLEARML_TASK_NAME="${CLEARML_TASK_NAME:-}"
CLEARML_TAGS="${CLEARML_TAGS:-}"

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

CLEARML_ARGS=()
if [[ -n "${CLEARML_PROJECT}" ]]; then
  CLEARML_ARGS+=(--clearml-project "${CLEARML_PROJECT}")
  [[ -n "${CLEARML_TASK_NAME}" ]] && CLEARML_ARGS+=(--clearml-task-name "${CLEARML_TASK_NAME}")
  [[ -n "${CLEARML_TAGS}" ]] && CLEARML_ARGS+=(--clearml-tags ${CLEARML_TAGS})
fi

"${PYTHON_BIN}" scripts/train_external_generators.py \
  --phase train \
  --models tabddpm \
  --targets ${TARGETS} \
  --python "${PYTHON_BIN}" \
  --device "${DEVICE}" \
  --tabddpm-steps "${TABDDPM_STEPS}" \
  "${CLEARML_ARGS[@]}"
