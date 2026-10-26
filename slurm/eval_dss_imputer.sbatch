#!/bin/bash
#SBATCH --job-name=dss_imputer_eval
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gpus=1

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-tabdiff}"
CONDA_SH="${CONDA_SH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-hdd_sequential}"
SPLIT="${SPLIT:-validation}"
SPLIT_LABEL="${SPLIT_LABEL:-${SPLIT}}"
PATTERNS="${PATTERNS:-mcar block loco}"
RATES="${RATES:-0.1 0.3 0.5}"
SAMPLES="${SAMPLES:-5}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
SAVE_SAMPLES="${SAVE_SAMPLES:-0}"
SAVE_FIRST_SAMPLE="${SAVE_FIRST_SAMPLE:-0}"
PLOT_AFTER_EVAL="${PLOT_AFTER_EVAL:-1}"
MODEL_LABEL="${MODEL_LABEL:-DSS Imputer}"
PLOT_SAMPLE_SIZE="${PLOT_SAMPLE_SIZE:-20000}"
MAX_DISCRETE_VALUES="${MAX_DISCRETE_VALUES:-50}"
GENERATIVE_METRICS="${GENERATIVE_METRICS:-0}"
METRIC_SAMPLE_SIZE="${METRIC_SAMPLE_SIZE:-5000}"
DCR_SAMPLE_SIZE="${DCR_SAMPLE_SIZE:-1000}"

case "${DATASET}" in
  hdd_sequential)
    TRAIN_CSV="dataset/pools/train_hdd_sequential.csv"
    TEST_CSV="dataset/pools/test_hdd_sequential.csv"
    DEFAULT_IOPS_LIMITS="0 14"
    DEFAULT_LAT_LIMITS="0 100"
    ;;
  ssd_sequential)
    TRAIN_CSV="dataset/pools/train_ssd_sequential.csv"
    TEST_CSV="dataset/pools/test_ssd_sequential.csv"
    DEFAULT_IOPS_LIMITS=""
    DEFAULT_LAT_LIMITS=""
    ;;
  ssd_random)
    TRAIN_CSV="dataset/pools/train_ssd_random.csv"
    TEST_CSV="dataset/pools/test_ssd_random.csv"
    DEFAULT_IOPS_LIMITS=""
    DEFAULT_LAT_LIMITS=""
    ;;
  cache)
    TRAIN_CSV="dataset/cache/train_cache.csv"
    TEST_CSV="dataset/cache/test_cache.csv"
    DEFAULT_IOPS_LIMITS=""
    DEFAULT_LAT_LIMITS=""
    ;;
  *) echo "Unknown DATASET=${DATASET}" >&2; exit 2 ;;
esac

case "${SPLIT}" in
  train) DEFAULT_DATA_CSV="${TRAIN_CSV}"; DEFAULT_TRAINING_SUBSET="fit" ;;
  validation) DEFAULT_DATA_CSV="${TEST_CSV}"; DEFAULT_TRAINING_SUBSET="validation" ;;
  test) DEFAULT_DATA_CSV="${TEST_CSV}"; DEFAULT_TRAINING_SUBSET="all" ;;
  *) echo "SPLIT must be train, validation, or test, got ${SPLIT}" >&2; exit 2 ;;
esac

CHECKPOINT="${CHECKPOINT:-trained_models/dss_imputer/${DATASET}${DATASET_CONFIG:+_clean}_full_train_val_test/best.pt}"
DATA_CSV="${DATA_CSV:-${DEFAULT_DATA_CSV}}"
TRAINING_SUBSET="${TRAINING_SUBSET:-${DEFAULT_TRAINING_SUBSET}}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/dss_imputer/${DATASET}${DATASET_CONFIG:+_clean}_full_train_val_test/evaluation_${SPLIT}}"
PLOTS_DIR="${PLOTS_DIR:-${OUTPUT_DIR}_plots}"
IOPS_LIMITS="${IOPS_LIMITS:-${DEFAULT_IOPS_LIMITS}}"
LAT_LIMITS="${LAT_LIMITS:-${DEFAULT_LAT_LIMITS}}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
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
set -u

if [[ -n "${DATASET_CONFIG:-}" ]]; then
  field=test_csv
  [[ "${SPLIT}" != "train" ]] || field=train_csv
  DATA_CSV=$("${PYTHON_BIN}" scripts/training_data.py --config "${DATASET_CONFIG}" --dataset "${DATASET}" --field "${field}")
fi

read -r -a PATTERN_ARRAY <<< "${PATTERNS}"
read -r -a RATE_ARRAY <<< "${RATES}"
COMMAND=(
  "${PYTHON_BIN}" scripts/eval_dss_imputer.py
  --data-csv "${DATA_CSV}"
  --split-label "${SPLIT_LABEL}"
  --training-subset "${TRAINING_SUBSET}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --patterns "${PATTERN_ARRAY[@]}"
  --rates "${RATE_ARRAY[@]}"
  --samples "${SAMPLES}"
  --batch-size "${BATCH_SIZE}"
  --metric-sample-size "${METRIC_SAMPLE_SIZE}"
  --dcr-sample-size "${DCR_SAMPLE_SIZE}"
)
if [[ "${GENERATIVE_METRICS}" == "1" ]]; then
  COMMAND+=(--generative-metrics)
fi
if [[ -n "${LOCO_COLS:-}" ]]; then
  read -r -a LOCO_ARRAY <<< "${LOCO_COLS}"
  COMMAND+=(--loco-cols "${LOCO_ARRAY[@]}")
fi
if [[ -n "${EXCLUDE_LOCO_COLS:-}" ]]; then
  read -r -a EXCLUDED_LOCO_ARRAY <<< "${EXCLUDE_LOCO_COLS}"
  COMMAND+=(--exclude-loco-cols "${EXCLUDED_LOCO_ARRAY[@]}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  COMMAND+=(--limit "${LIMIT}")
fi
if [[ "${SAVE_SAMPLES}" == "1" ]]; then
  COMMAND+=(--save-samples)
fi
if [[ "${SAVE_FIRST_SAMPLE}" == "1" ]]; then
  COMMAND+=(--save-first-sample)
fi

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"

if [[ "${PLOT_AFTER_EVAL}" == "1" ]]; then
  PLOT_COMMAND=(
    "${PYTHON_BIN}" scripts/plot_dss_imputer_evaluation.py
    --results-dir "${OUTPUT_DIR}"
    --data-csv "${DATA_CSV}"
    --output-dir "${PLOTS_DIR}"
    --model-label "${MODEL_LABEL}"
    --split-label "${SPLIT_LABEL}"
    --plot-sample-size "${PLOT_SAMPLE_SIZE}"
    --max-discrete-values "${MAX_DISCRETE_VALUES}"
  )
  if [[ -n "${IOPS_LIMITS}" ]]; then
    read -r -a IOPS_LIMIT_ARRAY <<< "${IOPS_LIMITS}"
    PLOT_COMMAND+=(--iops-limits "${IOPS_LIMIT_ARRAY[@]}")
  fi
  if [[ -n "${LAT_LIMITS}" ]]; then
    read -r -a LAT_LIMIT_ARRAY <<< "${LAT_LIMITS}"
    PLOT_COMMAND+=(--lat-limits "${LAT_LIMIT_ARRAY[@]}")
  fi
  printf 'Plot command:'
  printf ' %q' "${PLOT_COMMAND[@]}"
  printf '\n'
  "${PLOT_COMMAND[@]}"
fi
