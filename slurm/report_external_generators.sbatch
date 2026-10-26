#!/bin/bash
#SBATCH --job-name=dss_external_report
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky

set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
: "${MODEL:?Set MODEL}"
: "${DATANAME:?Set DATANAME}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
CONDA_ENV="${CONDA_ENV:-pytorch_new}"
cd "${PROJECT_DIR}"
mkdir -p slurm/logs
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LOKY_MAX_CPU_COUNT=1
if [[ -n "${CONDA_ENV}" ]]; then
  CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
  [[ -f "${CONDA_SH}" ]] || { echo "Missing conda activation script: ${CONDA_SH}" >&2; exit 2; }
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi
set -u
COMMAND=("${PYTHON_BIN:-python}" -u scripts/eval_external_generators.py
  --phase "${ACTION:-metrics}" --model "${MODEL}" --dataname "${DATANAME}"
  --output-dir "${OUTPUT_DIR}" --metric-sample-size "${METRIC_SAMPLE_SIZE:-5000}"
  --dcr-sample-size "${DCR_SAMPLE_SIZE:-1000}"
  --plot-sample-size "${PLOT_SAMPLE_SIZE:-0}"
  --pair-metric-sample-size "${PAIR_METRIC_SAMPLE_SIZE:-0}")
printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
"${COMMAND[@]}"
