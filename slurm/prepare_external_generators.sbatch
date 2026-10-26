#!/bin/bash
#SBATCH --job-name=dss_prepare_synth
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --partition=rocky

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV="${CONDA_ENV:-pytorch_new}"
CONDA_SH="${CONDA_SH:-}"
MODELS="${MODELS:-all}"
TARGETS="${TARGETS:-iops}"
REAL_TRAIN="${REAL_TRAIN:-dataset/pools/train_hdd_sequential.csv}"
REAL_TEST="${REAL_TEST:-dataset/pools/test_hdd_sequential.csv}"
DATASET_PREFIX="${DATASET_PREFIX:-dss_${DATASET:-hdd_sequential}${DATASET_CONFIG:+_clean}}"
SYNTH_DIR="${SYNTH_DIR:-synth_data}"
MODEL_DROP_COLS="${MODEL_DROP_COLS:-id}"
TABDDPM_STEPS="${TABDDPM_STEPS:-50000}"
TABDDPM_TIMESTEPS="${TABDDPM_TIMESTEPS:-1000}"
TABDDPM_BATCH_SIZE="${TABDDPM_BATCH_SIZE:-4096}"
TABDDPM_LR="${TABDDPM_LR:-0.0001}"
TABDDPM_D_LAYERS="${TABDDPM_D_LAYERS:-512 512 512 512}"
TABDDPM_DROPOUT="${TABDDPM_DROPOUT:-0.0}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV}"
set -u

DATASET_ARGS=()
if [[ -n "${DATASET_CONFIG:-}" ]]; then
  DATASET_ARGS=(--dataset-config "${DATASET_CONFIG}" --dataset "${DATASET:?Set DATASET with DATASET_CONFIG}")
fi

"${PYTHON_BIN}" scripts/train_external_generators.py \
  ${DATASET_ARGS[@]+"${DATASET_ARGS[@]}"} \
  --phase clone prepare \
  --models ${MODELS} \
  --targets ${TARGETS} \
  --real-train "${REAL_TRAIN}" \
  --real-test "${REAL_TEST}" \
  --dataset-prefix "${DATASET_PREFIX}" \
  --synth-dir "${SYNTH_DIR}" \
  --model-drop-cols ${MODEL_DROP_COLS} \
  --tabddpm-steps "${TABDDPM_STEPS}" \
  --tabddpm-timesteps "${TABDDPM_TIMESTEPS}" \
  --tabddpm-batch-size "${TABDDPM_BATCH_SIZE}" \
  --tabddpm-lr "${TABDDPM_LR}" \
  --tabddpm-d-layers ${TABDDPM_D_LAYERS} \
  --tabddpm-dropout "${TABDDPM_DROPOUT}"
