#!/bin/bash
#SBATCH --job-name=dss_tabddpm_train
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=rocky
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-tddpm}"
CONDA_SH="${CONDA_SH:-}"
TARGETS="${TARGETS:-iops}"
REAL_TRAIN="${REAL_TRAIN:-dataset/pools/train_hdd_sequential.csv}"
REAL_TEST="${REAL_TEST:-dataset/pools/test_hdd_sequential.csv}"
DATASET_PREFIX="${DATASET_PREFIX:-dss_${DATASET:-hdd_sequential}${DATASET_CONFIG:+_clean}}"
SYNTH_DIR="${SYNTH_DIR:-synth_data}"
DEVICE="${DEVICE:-cuda:0}"
TABDDPM_STEPS="${TABDDPM_STEPS:-50000}"
TABDDPM_TIMESTEPS="${TABDDPM_TIMESTEPS:-1000}"
TABDDPM_BATCH_SIZE="${TABDDPM_BATCH_SIZE:-4096}"
TABDDPM_LR="${TABDDPM_LR:-0.0001}"
TABDDPM_D_LAYERS="${TABDDPM_D_LAYERS:-512 512 512 512}"
TABDDPM_DROPOUT="${TABDDPM_DROPOUT:-0.0}"
MODEL_DROP_COLS="${MODEL_DROP_COLS:-id}"


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

DATASET_ARGS=()
if [[ -n "${DATASET_CONFIG:-}" ]]; then
  DATASET_ARGS=(--dataset-config "${DATASET_CONFIG}" --dataset "${DATASET:?Set DATASET with DATASET_CONFIG}")
fi

"${PYTHON_BIN}" scripts/train_external_generators.py \
  ${DATASET_ARGS[@]+"${DATASET_ARGS[@]}"} \
  --phase train \
  --models tabddpm \
  --targets ${TARGETS} \
  --real-train "${REAL_TRAIN}" \
  --real-test "${REAL_TEST}" \
  --dataset-prefix "${DATASET_PREFIX}" \
  --synth-dir "${SYNTH_DIR}" \
  --python "${PYTHON_BIN}" \
  --device "${DEVICE}" \
  --tabddpm-steps "${TABDDPM_STEPS}" \
  --tabddpm-timesteps "${TABDDPM_TIMESTEPS}" \
  --tabddpm-batch-size "${TABDDPM_BATCH_SIZE}" \
  --tabddpm-lr "${TABDDPM_LR}" \
  --tabddpm-d-layers ${TABDDPM_D_LAYERS} \
  --tabddpm-dropout "${TABDDPM_DROPOUT}" \
  --model-drop-cols ${MODEL_DROP_COLS}
