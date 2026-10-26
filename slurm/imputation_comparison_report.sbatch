#!/bin/bash
#SBATCH --job-name=dss_imputation_report
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-pytorch_new}"
CONDA_SH="${CONDA_SH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-external/tabdiff/data}"
TABDIFF_ROOT="${TABDIFF_ROOT:-reports/imputation}"
TABSYN_ROOT="${TABSYN_ROOT:-reports/imputation_tabsyn}"
DSS_RAW_ROOT="${DSS_RAW_ROOT:-}"
DSS_LOG_ROOT="${DSS_LOG_ROOT:-}"
DSS_MODEL_SPECS="${DSS_MODEL_SPECS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/imputation_tabdiff_tabsyn_comparison}"
DATASETS="${DATASETS:-dss_hdd_sequential_iops dss_ssd_sequential_iops dss_ssd_random_iops dss_cache_iops}"
MODELS="${MODELS:-tabdiff tabsyn}"
TARGETS="${TARGETS:-all}"
METRIC_SAMPLE_SIZE="${METRIC_SAMPLE_SIZE:-5000}"
METRIC_NUMERICAL_COLS="${METRIC_NUMERICAL_COLS:-}"
DCR_SAMPLE_SIZE="${DCR_SAMPLE_SIZE:-1000}"
ENERGY_SAMPLE_SIZE="${ENERGY_SAMPLE_SIZE:-1000}"
PLOT_SAMPLE_SIZE="${PLOT_SAMPLE_SIZE:-20000}"

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
  set +u
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  set -u
fi

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a MODEL_ARRAY <<< "${MODELS}"
read -r -a TARGET_ARRAY <<< "${TARGETS}"

COMMAND=(
  "${PYTHON_BIN}" scripts/imputation_comparison_report.py
  --data-root "${DATA_ROOT}"
  --tabdiff-root "${TABDIFF_ROOT}"
  --tabsyn-root "${TABSYN_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --datasets "${DATASET_ARRAY[@]}"
  --models "${MODEL_ARRAY[@]}"
  --targets "${TARGET_ARRAY[@]}"
  --metric-sample-size "${METRIC_SAMPLE_SIZE}"
  --dcr-sample-size "${DCR_SAMPLE_SIZE}"
  --energy-sample-size "${ENERGY_SAMPLE_SIZE}"
  --plot-sample-size "${PLOT_SAMPLE_SIZE}"
)
if [[ -n "${METRIC_NUMERICAL_COLS}" ]]; then
  read -r -a METRIC_NUMERICAL_ARRAY <<< "${METRIC_NUMERICAL_COLS}"
  COMMAND+=(--numerical-cols "${METRIC_NUMERICAL_ARRAY[@]}")
fi
if [[ -n "${DSS_RAW_ROOT}" ]]; then
  COMMAND+=(--dss-raw-root "${DSS_RAW_ROOT}")
fi
if [[ -n "${DSS_LOG_ROOT}" ]]; then
  COMMAND+=(--dss-log-root "${DSS_LOG_ROOT}")
fi
if [[ -n "${DSS_MODEL_SPECS}" ]]; then
  IFS=';' read -r -a DSS_SPEC_ARRAY <<< "${DSS_MODEL_SPECS}"
  for specification in "${DSS_SPEC_ARRAY[@]}"; do
    IFS='|' read -r key label root <<< "${specification}"
    if [[ -z "${key}" || -z "${label}" || -z "${root}" ]]; then
      echo "Invalid DSS_MODEL_SPECS entry: ${specification}" >&2
      exit 2
    fi
    COMMAND+=(--dss-model "${key}" "${label}" "${root}")
  done
fi

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
