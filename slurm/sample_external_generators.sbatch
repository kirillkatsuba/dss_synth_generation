#!/bin/bash
#SBATCH --job-name=dss_external_sample
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gpus=1

set -eo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
: "${MODEL:?Set MODEL to tabsyn or tabdiff}"
: "${DATANAME:?Set DATANAME}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
CONDA_ENV="${CONDA_ENV:-${MODEL}}"
cd "${PROJECT_DIR}"
mkdir -p slurm/logs
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
if [[ -n "${CONDA_ENV}" ]]; then
  CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
  [[ -f "${CONDA_SH}" ]] || { echo "Missing conda activation script: ${CONDA_SH}" >&2; exit 2; }
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi
set -u
COMMAND=("${PYTHON_BIN:-python}" -u scripts/eval_external_generators.py
  --phase sample --model "${MODEL}" --dataname "${DATANAME}"
  --output-dir "${OUTPUT_DIR}" --protocol "${PROTOCOL:-unconditional}"
  --split "${SPLIT:-test}" --trials "${TRIALS:-5}"
  --batch-size "${BATCH_SIZE:-2048}" --steps "${STEPS:-50}"
  --resample-rounds "${RESAMPLE_ROUNDS:-1}"
  --impute-condition "${IMPUTE_CONDITION:-x_t}" --seed "${SEED:-42}" --gpu "${GPU:-0}")
[[ -z "${TABDIFF_EXP_NAME:-}" ]] || COMMAND+=(--exp-name "${TABDIFF_EXP_NAME}")
printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
"${COMMAND[@]}"
