#!/bin/bash
#SBATCH --job-name=dss_imputer_plots
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-tabdiff}"
CONDA_SH="${CONDA_SH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-hdd_sequential}"
MODEL_DIR="${MODEL_DIR:-trained_models/dss_imputer/${DATASET}_full_train_val_test}"
EVALUATION_DIR="${EVALUATION_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_DIR}/plots}"
TARGETS="${TARGETS:-iops lat}"
SAMPLE_SIZE="${SAMPLE_SIZE:-20000}"

case "${DATASET}" in
  hdd_sequential) TEST_CSV="dataset/pools/test_hdd_sequential.csv" ;;
  ssd_sequential) TEST_CSV="dataset/pools/test_ssd_sequential.csv" ;;
  ssd_random) TEST_CSV="dataset/pools/test_ssd_random.csv" ;;
  cache) TEST_CSV="dataset/cache/test_cache.csv" ;;
  *) echo "Unknown DATASET=${DATASET}" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
mkdir -p slurm/logs "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1

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

read -r -a TARGET_ARRAY <<< "${TARGETS}"
COMMAND=(
  "${PYTHON_BIN}" scripts/plot_dss_imputer.py
  --model-dir "${MODEL_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --targets "${TARGET_ARRAY[@]}"
  --sample-size "${SAMPLE_SIZE}"
)
if [[ -n "${EVALUATION_DIR}" ]]; then
  COMMAND+=(
    --evaluation-dir "${EVALUATION_DIR}"
    --test-csv "${TEST_CSV}"
  )
fi

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
