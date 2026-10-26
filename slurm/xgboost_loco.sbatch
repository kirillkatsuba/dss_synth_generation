#!/bin/bash
#SBATCH --job-name=dss_xgb_loco
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-tabdiff}"
CONDA_SH="${CONDA_SH:-}"
DATANAME="${DATANAME:?DATANAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/imputation/${DATANAME}/xgboost_loco}"
TARGET_COLUMNS="${TARGET_COLUMNS:-all}"
DEVICE="${DEVICE:-cuda}"
AUTHORS_TARGET_TRANSFORM="${AUTHORS_TARGET_TRANSFORM:-0}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs

if [[ -z "${CONDA_SH}" ]]; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

EXTRA_ARGS=()
if [[ "${AUTHORS_TARGET_TRANSFORM}" == "1" ]]; then
  EXTRA_ARGS+=(--authors-target-transform)
fi

python scripts/xgboost_loco.py \
  --dataname "${DATANAME}" \
  --tabdiff-repo external/tabdiff \
  --output-dir "${OUTPUT_DIR}" \
  --columns ${TARGET_COLUMNS} \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
