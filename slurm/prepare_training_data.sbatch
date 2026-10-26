#!/bin/bash
#SBATCH --job-name=dss_clean_data
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky

set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-pytorch_new}"
CONDA_SH="${CONDA_SH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-hdd_sequential ssd_sequential ssd_random cache}"
SOURCE_CONFIG="${SOURCE_CONFIG:-configs/dss_imputer/datasets.json}"
CLEAN_DATA_DIR="${CLEAN_DATA_DIR:-dataset/cleaned}"
OUTLIER_CONTAMINATION="${OUTLIER_CONTAMINATION:-0.05}"
OUTLIER_SEED="${OUTLIER_SEED:-42}"
OUTLIER_EPSILON="${OUTLIER_EPSILON:-1e-6}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -n "${CONDA_ENV}" ]]; then
  CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
  [[ -f "${CONDA_SH}" ]] || { echo "Conda script not found: ${CONDA_SH}" >&2; exit 2; }
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi
set -u
read -r -a DATASET_ARRAY <<< "${DATASETS}"
COMMAND=("${PYTHON_BIN}" scripts/clean_dataset_outliers.py
  --config "${SOURCE_CONFIG}" --datasets "${DATASET_ARRAY[@]}"
  --output-dir "${CLEAN_DATA_DIR}" --prepared-config "${CLEAN_DATA_DIR}/datasets.json"
  --contamination "${OUTLIER_CONTAMINATION}" --seed "${OUTLIER_SEED}" --epsilon "${OUTLIER_EPSILON}")
[[ "${OVERWRITE:-0}" != "1" ]] || COMMAND+=(--overwrite)
printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
"${COMMAND[@]}"
