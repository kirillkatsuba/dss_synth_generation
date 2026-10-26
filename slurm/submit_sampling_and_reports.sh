#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-}"
if [[ "${1:-}" == "-A" ]]; then
  ACCOUNT="${2:-}"
elif [[ -n "${1:-}" ]]; then
  ACCOUNT="${1}"
fi

if [[ -z "${ACCOUNT}" ]]; then
  echo "Usage: ACCOUNT=proj_1808 bash slurm/submit_sampling_and_reports.sh"
  echo "   or: bash slurm/submit_sampling_and_reports.sh -A proj_1808"
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
TARGETS="${TARGETS:-iops}"
TARGET_COLS="${TARGET_COLS:-iops lat}"
MODEL_DROP_COLS="${MODEL_DROP_COLS:-id}"
METRIC_ENV="${METRIC_ENV:-pytorch_new}"
TABDDPM_ENV="${TABDDPM_ENV:-tddpm}"
TABDIFF_ENV="${TABDIFF_ENV:-tabdiff}"
TABSYN_ENV="${TABSYN_ENV:-tabsyn}"
MODELS="${MODELS:-tabddpm tabdiff tabsyn}"
DATASET_SUFFIX="${DATASET_SUFFIX:-${DATASET_CONFIG:+_clean}}"

# Datasets trained after the original hdd_sequential run.
DATASETS="${DATASETS:-ssd_random ssd_sequential cache}"

for model in ${MODELS}; do
  case "${model}" in
    tabddpm|tabdiff|tabsyn) ;;
    *) echo "Unknown model: ${model}" >&2; exit 2 ;;
  esac
done

dataset_paths() {
  local dataset="$1"
  case "${dataset}" in
    hdd_sequential)
      echo "dataset/pools/train_hdd_sequential.csv dataset/pools/test_hdd_sequential.csv dss_hdd_sequential"
      ;;
    ssd_random)
      echo "dataset/pools/train_ssd_random.csv dataset/pools/test_ssd_random.csv dss_ssd_random"
      ;;
    ssd_sequential)
      echo "dataset/pools/train_ssd_sequential.csv dataset/pools/test_ssd_sequential.csv dss_ssd_sequential"
      ;;
    cache)
      echo "dataset/cache/train_cache.csv dataset/cache/test_cache.csv dss_cache"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      exit 2
      ;;
  esac
}

submit_metric_job() {
  local dependency="$1"
  local real_train="$2"
  local real_test="$3"
  local synthetic="$4"
  local output_dir="$5"
  local label="$6"

  sbatch --parsable -A "${ACCOUNT}" --partition=rocky --dependency=afterok:"${dependency}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONDA_ENV=${METRIC_ENV},REAL_TRAIN=${real_train},REAL_TEST=${real_test},SYNTHETIC=${synthetic},OUTPUT_DIR=${output_dir},SYNTHETIC_LABEL=${label},TARGET_COLS=${TARGET_COLS},MODEL_DROP_COLS=${MODEL_DROP_COLS}" \
    slurm/metric_eval.sbatch
}

cd "${PROJECT_DIR}"

for dataset in ${DATASETS}; do
  read -r real_train real_test dataset_prefix < <(dataset_paths "${dataset}")
  if [[ -n "${DATASET_CONFIG:-}" ]]; then
    real_train="$("${CONFIG_PYTHON:-python3}" scripts/training_data.py --config "${DATASET_CONFIG}" --dataset "${dataset}" --field train_csv)"
    real_test="$("${CONFIG_PYTHON:-python3}" scripts/training_data.py --config "${DATASET_CONFIG}" --dataset "${dataset}" --field test_csv)"
  fi
  dataset_prefix="${dataset_prefix}${DATASET_SUFFIX}"
  synth_dir="synth_data/${dataset_prefix}"

  common_export="ALL,PROJECT_DIR=${PROJECT_DIR},TARGETS=${TARGETS},REAL_TRAIN=${real_train},REAL_TEST=${real_test},DATASET_PREFIX=${dataset_prefix},SYNTH_DIR=${synth_dir},MODEL_DROP_COLS=${MODEL_DROP_COLS}"

  for model in ${MODELS}; do
    case "${model}" in
      tabddpm) conda_env="${TABDDPM_ENV}" ;;
      tabdiff) conda_env="${TABDIFF_ENV}" ;;
      tabsyn) conda_env="${TABSYN_ENV}" ;;
    esac
    sample_job=$(
      sbatch --parsable -A "${ACCOUNT}" --partition=rocky \
        --export="${common_export},CONDA_ENV=${conda_env},TABDIFF_EXP_NAME=dss_tabdiff_${dataset_prefix}" \
        "slurm/sample_${model}.sbatch"
    )
    sample_job="${sample_job%%;*}"
    metric_job=$(
      submit_metric_job "${sample_job}" "${real_train}" "${real_test}" \
        "${synth_dir}/${model}_${dataset_prefix}_iops.csv" \
        "reports/metric_eval_${dataset_prefix}_${model}" "${model}"
    )
    echo "${dataset}: ${model} sample ${sample_job}, metric ${metric_job}"
  done
done
