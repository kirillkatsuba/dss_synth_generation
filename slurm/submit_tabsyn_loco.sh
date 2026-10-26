#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-}"
if [[ "${1:-}" == "-A" ]]; then
  ACCOUNT="${2:-}"
elif [[ -n "${1:-}" ]]; then
  ACCOUNT="${1}"
fi

if [[ -z "${ACCOUNT}" ]]; then
  echo "Usage: bash slurm/submit_tabsyn_loco.sh -A proj_1808"
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-tabsyn}"
DATANAME="${DATANAME:-dss_hdd_sequential_iops}"
TRAIN_CSV="${TRAIN_CSV:-external/tabsyn/data/${DATANAME}/train.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reports/imputation_tabsyn/${DATANAME}/loco}"
TRIALS="${TRIALS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
STEPS="${STEPS:-50}"
RESAMPLE_ROUNDS="${RESAMPLE_ROUNDS:-1}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:-}"
EXCLUDE_COLS="${EXCLUDE_COLS:-}"

cd "${PROJECT_DIR}"
if [[ ! -f "${TRAIN_CSV}" ]]; then
  echo "Prepared TabSyn train CSV not found: ${TRAIN_CSV}" >&2
  exit 2
fi

list_columns_cmd=(python3 scripts/list_imputation_columns.py --csv "${TRAIN_CSV}")
if [[ -n "${EXCLUDE_COLS}" ]]; then
  read -r -a excluded <<< "${EXCLUDE_COLS}"
  list_columns_cmd+=(--exclude "${excluded[@]}")
fi

columns=("__placeholder__")
while IFS= read -r column; do
  columns+=("${column}")
done < <("${list_columns_cmd[@]}")
columns=("${columns[@]:1}")

if [[ "${#columns[@]}" -eq 0 ]]; then
  echo "No non-constant columns found in ${TRAIN_CSV}" >&2
  exit 2
fi

echo "Submitting ${#columns[@]} TabSyn LOCO jobs for ${DATANAME}: ${columns[*]}"
for column in "${columns[@]}"; do
  output_name="${column//[^a-zA-Z0-9_.-]/_}"
  export_values="ALL,PROJECT_DIR=${PROJECT_DIR},CONDA_ENV=${CONDA_ENV},DATANAME=${DATANAME},MASK_COLS=${column},OUTPUT_DIR=${OUTPUT_ROOT}/${output_name},TRIALS=${TRIALS},BATCH_SIZE=${BATCH_SIZE},STEPS=${STEPS},RESAMPLE_ROUNDS=${RESAMPLE_ROUNDS},GPU=${GPU}"
  if [[ -n "${LIMIT}" ]]; then
    export_values+=",LIMIT=${LIMIT}"
  fi
  job_id=$(sbatch --parsable -A "${ACCOUNT}" \
    --job-name="tabsyn_${output_name}" \
    --export="${export_values}" \
    slurm/impute_tabsyn.sbatch)
  echo "${column}: job ${job_id} -> ${OUTPUT_ROOT}/${output_name}"
done
