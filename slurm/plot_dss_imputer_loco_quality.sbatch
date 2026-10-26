#!/bin/bash
#SBATCH --job-name=dss_loco_quality
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-pytorch_new}"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${RESULTS_DIR:?Set RESULTS_DIR to the completed evaluation directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}/quality_plots}"
PLOT_SAMPLE_SIZE="${PLOT_SAMPLE_SIZE:-20000}"
METRIC_SAMPLE_SIZE="${METRIC_SAMPLE_SIZE:-20000}"
PREDICTION_SOURCE="${PREDICTION_SOURCE:-trial_000}"
PAIR_PLOT_KIND="${PAIR_PLOT_KIND:-auto}"
MAX_CATEGORIES="${MAX_CATEGORIES:-12}"
case "${DATASET:-hdd_sequential}" in
  hdd_sequential) DEFAULT_IOPS_LIMITS="0 14"; DEFAULT_LAT_LIMITS="0 100" ;;
  ssd_sequential|ssd_random|cache) DEFAULT_IOPS_LIMITS=""; DEFAULT_LAT_LIMITS="" ;;
  *) echo "Unknown DATASET=${DATASET}" >&2; exit 2 ;;
esac
IOPS_LIMITS="${IOPS_LIMITS-${DEFAULT_IOPS_LIMITS}}"
LAT_LIMITS="${LAT_LIMITS-${DEFAULT_LAT_LIMITS}}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -n "${CONDA_ENV}" ]]; then
  if [[ -z "${CONDA_SH:-}" ]]; then
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
      CONDA_BASE="$("${CONDA_EXE}" info --base)"
    else
      CONDA_BASE="$(conda info --base)"
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

COMMAND=(
  "${PYTHON_BIN}" scripts/plot_dss_imputer_loco_quality.py
  --results-dir "${RESULTS_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --prediction-source "${PREDICTION_SOURCE}"
  --plot-sample-size "${PLOT_SAMPLE_SIZE}"
  --metric-sample-size "${METRIC_SAMPLE_SIZE}"
  --pair-plot-kind "${PAIR_PLOT_KIND}"
  --max-categories "${MAX_CATEGORIES}"
)
if [[ -n "${IOPS_LIMITS}" ]]; then
  read -r -a limits <<< "${IOPS_LIMITS}"
  COMMAND+=(--iops-limits "${limits[@]}")
fi
if [[ -n "${LAT_LIMITS}" ]]; then
  read -r -a limits <<< "${LAT_LIMITS}"
  COMMAND+=(--lat-limits "${limits[@]}")
fi
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
