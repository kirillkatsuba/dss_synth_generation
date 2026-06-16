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
CLEARML_PROJECT="${CLEARML_PROJECT:-}"
CLEARML_TASK_NAME="${CLEARML_TASK_NAME:-}"
CLEARML_TAGS="${CLEARML_TAGS:-}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

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
