#!/bin/bash
#SBATCH --job-name=dss_imputer_sample
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gpus=1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-tabdiff}"
CONDA_SH="${CONDA_SH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-hdd_sequential}"
SAMPLES="${SAMPLES:-10}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
MASK_COLS="${MASK_COLS:-}"

case "${DATASET}" in
  hdd_sequential) DEFAULT_INPUT="dataset/pools/test_hdd_sequential.csv" ;;
  ssd_sequential) DEFAULT_INPUT="dataset/pools/test_ssd_sequential.csv" ;;
  ssd_random) DEFAULT_INPUT="dataset/pools/test_ssd_random.csv" ;;
  cache) DEFAULT_INPUT="dataset/cache/test_cache.csv" ;;
  *) echo "Unknown DATASET=${DATASET}" >&2; exit 2 ;;
esac

INPUT_CSV="${INPUT_CSV:-${DEFAULT_INPUT}}"
CHECKPOINT="${CHECKPOINT:-trained_models/dss_imputer/${DATASET}_full_train_val_test/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/imputation/dss_imputer/${DATASET}_full_train_val_test/sample}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs "${OUTPUT_DIR}"
if [[ -n "${CONDA_ENV}" ]]; then
  if [[ -z "${CONDA_SH}" ]]; then
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
      CONDA_BASE="$("${CONDA_EXE}" info --base)"
    elif command -v conda >/dev/null 2>&1; then
      CONDA_BASE="$(conda info --base)"
    else
      echo "Cannot locate Conda. Pass CONDA_SH=/path/to/etc/profile.d/conda.sh" >&2
      exit 2
    fi
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda activation script does not exist: ${CONDA_SH}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

COMMAND=(
  "${PYTHON_BIN}" scripts/impute_dss_imputer.py
  --input-csv "${INPUT_CSV}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --samples "${SAMPLES}"
  --batch-size "${BATCH_SIZE}"
)
if [[ -n "${MASK_COLS}" ]]; then
  read -r -a MASK_ARRAY <<< "${MASK_COLS}"
  COMMAND+=(--mask-cols "${MASK_ARRAY[@]}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  COMMAND+=(--limit "${LIMIT}")
fi

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
