#!/bin/bash
#SBATCH --job-name=dss_tabdiff_impute
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-tabdiff}"
CONDA_SH="${CONDA_SH:-}"
DATANAME="${DATANAME:-dss_hdd_sequential_iops}"
TABDIFF_EXP_NAME="${TABDIFF_EXP_NAME:-}"
MASK_COLS="${MASK_COLS:-iops}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/imputation/${DATANAME}}"
TRIALS="${TRIALS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
RESAMPLE_ROUNDS="${RESAMPLE_ROUNDS:-1}"
IMPUTE_CONDITION="${IMPUTE_CONDITION:-x_t}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:-}"

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

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

EXP_ARGS=()
if [[ -n "${TABDIFF_EXP_NAME}" ]]; then
  EXP_ARGS+=(--exp-name "${TABDIFF_EXP_NAME}")
fi

"${PYTHON_BIN}" scripts/tabdiff_impute.py \
  --dataname "${DATANAME}" \
  "${EXP_ARGS[@]}" \
  --mask-cols ${MASK_COLS} \
  --output-dir "${OUTPUT_DIR}" \
  --trials "${TRIALS}" \
  --batch-size "${BATCH_SIZE}" \
  --resample-rounds "${RESAMPLE_ROUNDS}" \
  --impute-condition "${IMPUTE_CONDITION}" \
  --gpu "${GPU}" \
  "${LIMIT_ARGS[@]}"
