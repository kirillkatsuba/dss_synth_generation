#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-}"
if [[ "${1:-}" == "-A" ]]; then
  ACCOUNT="${2:-}"
elif [[ -n "${1:-}" ]]; then
  ACCOUNT="${1}"
fi

if [[ -z "${ACCOUNT}" ]]; then
  echo "Usage: ACCOUNT=proj_1808 bash slurm/submit_remaining_datasets.sh"
  echo "   or: bash slurm/submit_remaining_datasets.sh -A proj_1808"
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
TARGETS="${TARGETS:-iops}"
MODEL_DROP_COLS="${MODEL_DROP_COLS:-id}"
PREPARE_ENV="${PREPARE_ENV:-pytorch_new}"
TABDDPM_ENV="${TABDDPM_ENV:-tddpm}"
TABDIFF_ENV="${TABDIFF_ENV:-tabdiff}"
TABSYN_ENV="${TABSYN_ENV:-tabsyn}"
DATASET_SUFFIX="${DATASET_SUFFIX:-${DATASET_CONFIG:+_clean}}"

# Default is the datasets that were not part of the first hdd_sequential run.
DATASETS="${DATASETS:-ssd_random ssd_sequential cache}"

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

cd "${PROJECT_DIR}"

for dataset in ${DATASETS}; do
  read -r real_train real_test dataset_prefix < <(dataset_paths "${dataset}")
  dataset_prefix="${dataset_prefix}${DATASET_SUFFIX}"
  synth_dir="synth_data/${dataset_prefix}"
  tabdiff_exp_name="dss_tabdiff_${dataset_prefix}"

  common_export="ALL,DATASET=${dataset},TARGETS=${TARGETS},REAL_TRAIN=${real_train},REAL_TEST=${real_test},DATASET_PREFIX=${dataset_prefix},SYNTH_DIR=${synth_dir},MODEL_DROP_COLS=${MODEL_DROP_COLS}"

  prepare_job=$(
    sbatch --parsable -A "${ACCOUNT}" \
      --export="${common_export},CONDA_ENV=${PREPARE_ENV},MODELS=all" \
      slurm/prepare_external_generators.sbatch
  )
  echo "${dataset}: prepare job ${prepare_job}"

  tabddpm_job=$(
    sbatch --parsable -A "${ACCOUNT}" --dependency=afterok:"${prepare_job}" \
      --export="${common_export},CONDA_ENV=${TABDDPM_ENV}" \
      slurm/train_tabddpm.sbatch
  )
  echo "${dataset}: tabddpm train job ${tabddpm_job}"

  tabdiff_job=$(
    sbatch --parsable -A "${ACCOUNT}" --dependency=afterok:"${prepare_job}" \
      --export="${common_export},CONDA_ENV=${TABDIFF_ENV},TABDIFF_EXP_NAME=${tabdiff_exp_name}" \
      slurm/train_tabdiff.sbatch
  )
  echo "${dataset}: tabdiff train job ${tabdiff_job}"

  tabsyn_job=$(
    sbatch --parsable -A "${ACCOUNT}" --dependency=afterok:"${prepare_job}" \
      --export="${common_export},CONDA_ENV=${TABSYN_ENV}" \
      slurm/train_tabsyn.sbatch
  )
  echo "${dataset}: tabsyn train job ${tabsyn_job}"
done
