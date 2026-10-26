#!/bin/bash
#SBATCH --job-name=dss_metric_eval
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=rocky

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-pytorch_new}"
CONDA_SH="${CONDA_SH:-}"
REAL_TRAIN="${REAL_TRAIN:?REAL_TRAIN is required}"
REAL_TEST="${REAL_TEST:?REAL_TEST is required}"
SYNTHETIC="${SYNTHETIC:?SYNTHETIC is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
SYNTHETIC_LABEL="${SYNTHETIC_LABEL:?SYNTHETIC_LABEL is required}"
TARGET_COLS="${TARGET_COLS:-iops lat}"
MODEL_DROP_COLS="${MODEL_DROP_COLS:-id}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

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
set -u

"${PYTHON_BIN}" scripts/metric_eval.py \
  --real-train "${REAL_TRAIN}" \
  --real-test "${REAL_TEST}" \
  --synthetic "${SYNTHETIC}" \
  --output-dir "${OUTPUT_DIR}" \
  --synthetic-label "${SYNTHETIC_LABEL}" \
  --target-cols ${TARGET_COLS} \
  --model-drop-cols ${MODEL_DROP_COLS}
