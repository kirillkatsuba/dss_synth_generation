#!/bin/bash
#SBATCH --job-name=dss_prepare_synth
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/dss_synth_generation}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-}"
CONDA_SH="${CONDA_SH:-}"
MODELS="${MODELS:-all}"
TARGETS="${TARGETS:-iops}"
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
  --phase clone prepare \
  --models ${MODELS} \
  --targets ${TARGETS} \
  "${CLEARML_ARGS[@]}"
