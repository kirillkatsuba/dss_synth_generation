#!/bin/bash
#SBATCH --job-name=dss_tabdiff_train
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-}"
CONDA_SH="${CONDA_SH:-}"
TARGETS="${TARGETS:-iops}"
GPU="${GPU:-0}"
TABDIFF_EXP_NAME="${TABDIFF_EXP_NAME:-dss_tabdiff}"
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
  --models tabdiff \
  --targets ${TARGETS} \
  --python "${PYTHON_BIN}" \
  --gpu "${GPU}" \
  --tabdiff-exp-name "${TABDIFF_EXP_NAME}" \
  "${CLEARML_ARGS[@]}"
