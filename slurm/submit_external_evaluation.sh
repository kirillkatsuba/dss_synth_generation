#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-proj_1808}"
if [[ "${1:-}" == "-A" ]]; then ACCOUNT="${2:?Missing account}";
elif [[ -n "${1:-}" ]]; then ACCOUNT="$1"; fi
PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
PHASE="${PHASE:-unconditional}"
DATASETS="${DATASETS:-hdd_sequential ssd_sequential ssd_random cache}"
MODELS="${MODELS:-tabsyn tabdiff}"
RUN_TAG="${RUN_TAG:-uncond_all_20261015}"
TABDIFF_EXP_NAME_TEMPLATE="${TABDIFF_EXP_NAME_TEMPLATE:-}"
if [[ -z "${TABDIFF_EXP_NAME_TEMPLATE}" ]]; then
  TABDIFF_EXP_NAME_TEMPLATE='tabdiff_{dataset}_{run_tag}'
fi
REPORT_ROOT="${REPORT_ROOT:-reports/external_generators/${RUN_TAG}}"
METRICS_ENV="${METRICS_ENV:-pytorch_new}"
SAMPLE_TIME="${SAMPLE_TIME:-05:00:00}"
METRIC_TIME="${METRIC_TIME:-05:00:00}"
PLOT_TIME="${PLOT_TIME:-00:15:00}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_DELAY="${SUBMIT_DELAY:-1}"
JOB_MANIFEST="${JOB_MANIFEST:-slurm/logs/external_eval_$(date +%Y%m%d_%H%M%S)_$$.tsv}"
case "${PHASE}" in unconditional|loco|all|reports) ;; *) echo "Unknown PHASE=${PHASE}" >&2; exit 2 ;; esac
cd "${PROJECT_DIR}"

tabdiff_exp_name() {
  local name="${TABDIFF_EXP_NAME_TEMPLATE}"
  name="${name//\{dataset\}/$1}"
  printf '%s\n' "${name//\{run_tag\}/${RUN_TAG}}"
}

# Check the complete request before submitting any work.
shopt -s nullglob
for dataset in ${DATASETS}; do
  case "${dataset}" in hdd_sequential|ssd_sequential|ssd_random|cache) ;; *) echo "Unknown dataset: ${dataset}" >&2; exit 2 ;; esac
  dataname="dss_${dataset}_${RUN_TAG}_iops"
  for model in ${MODELS}; do
    case "${model}" in tabsyn|tabdiff) ;; *) echo "Unknown model: ${model}" >&2; exit 2 ;; esac
    repo="external/${model}"
    for file in "${repo}/data/${dataname}/info.json" "${repo}/data/${dataname}/train.csv" "${repo}/data/${dataname}/test.csv"; do
      [[ -f "${file}" ]] || { echo "Missing prepared data: ${file}" >&2; exit 2; }
    done
    if [[ "${PHASE}" != reports ]]; then
      if [[ "${model}" == tabsyn ]]; then
        for file in "${repo}/tabsyn/ckpt/${dataname}/model.pt" "${repo}/tabsyn/vae/ckpt/${dataname}/model.pt" "${repo}/tabsyn/vae/ckpt/${dataname}/train_z.npy"; do
          [[ -f "${file}" ]] || { echo "Missing trained TabSyn artifact: ${file}" >&2; exit 2; }
        done
      else
        run="${repo}/tabdiff/ckpt/${dataname}/$(tabdiff_exp_name "${dataset}")"
        checkpoints=("${run}"/best_ema_model*.pt "${run}"/best_model*.pt "${run}"/model_*.pt)
        [[ -f "${run}/config.pkl" && "${#checkpoints[@]}" -gt 0 ]] || { echo "Missing trained TabDiff checkpoint/config: ${run}" >&2; exit 2; }
      fi
    fi
    if [[ "${PHASE}" == reports ]]; then
      found=0
      for split in unconditional loco_train loco_test; do
        [[ ! -f "${REPORT_ROOT}/${model}/${dataset}/${split}/run_info.json" ]] || found=1
      done
      [[ "${found}" == 1 ]] || { echo "No saved results for ${model}/${dataset}" >&2; exit 2; }
    else
      requested=()
      if [[ "${PHASE}" == unconditional || "${PHASE}" == all ]]; then requested+=(unconditional); fi
      if [[ "${PHASE}" == loco || "${PHASE}" == all ]]; then requested+=(loco_train loco_test); fi
      for split in "${requested[@]}"; do
        saved=("${REPORT_ROOT}/${model}/${dataset}/${split}"/*/trial_*.csv)
        [[ "${#saved[@]}" == 0 ]] || { echo "Saved trials already exist: ${REPORT_ROOT}/${model}/${dataset}/${split}. Use PHASE=reports to reuse them." >&2; exit 2; }
      done
    fi
  done
done

if [[ "${DRY_RUN}" != 1 ]]; then
  mkdir -p "$(dirname "${JOB_MANIFEST}")" slurm/logs
  [[ ! -e "${JOB_MANIFEST}" ]] || { echo "Manifest already exists: ${JOB_MANIFEST}" >&2; exit 2; }
  printf 'dataset\tmodel\tprotocol\tsplit\taction\tjob_id\tdependency\toutput_dir\n' > "${JOB_MANIFEST}"
fi
counter=0
submit_job() {
  local script="$1" dependency="$2" time_limit="$3" response
  local -a command=(sbatch --parsable -A "${ACCOUNT}" --partition=rocky --nodes=1 --cpus-per-task=4
                    "--time=${time_limit}" "--job-name=${DATASET}_${MODEL}_${PROTOCOL}_${SPLIT}_${ACTION}" --export=ALL)
  [[ -z "${dependency}" ]] || command+=("--dependency=afterok:${dependency}")
  command+=("${script}")
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'DRY RUN: MODEL=%q CONDA_ENV=%q DATANAME=%q OUTPUT_DIR=%q ACTION=%q ' "${MODEL}" "${CONDA_ENV}" "${DATANAME}" "${OUTPUT_DIR}" "${ACTION}"
    printf '%q ' "${command[@]}"; printf '\n'
    counter=$((counter + 1)); LAST_JOB="dry${counter}"
    return
  fi
  response="$("${command[@]}")" || { echo "Submission failed: ${MODEL}/${DATASET}/${ACTION}" >&2; exit 1; }
  LAST_JOB="${response%%;*}"
  [[ "${LAST_JOB}" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch job id: ${response}" >&2; exit 1; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${DATASET}" "${MODEL}" "${PROTOCOL}" "${SPLIT}" "${ACTION}" "${LAST_JOB}" "${dependency}" "${OUTPUT_DIR}" >> "${JOB_MANIFEST}"
  echo "${MODEL}/${DATASET}/${PROTOCOL}/${SPLIT}: ${ACTION} job ${LAST_JOB}"
  sleep "${SUBMIT_DELAY}"
}

submit_protocol() {
  local protocol="$1" split="$2" dependency="$3" suffix="${1}"
  [[ "${protocol}" != loco ]] || suffix="loco_${split}"
  export PROTOCOL="${protocol}" SPLIT="${split}"
  export OUTPUT_DIR="${REPORT_ROOT}/${MODEL}/${DATASET}/${suffix}"
  if [[ "${PHASE}" != reports ]]; then
    export CONDA_ENV="${MODEL}" ACTION=sample
    submit_job slurm/sample_external_generators.sbatch "${dependency}" "${SAMPLE_TIME}"
    dependency="${LAST_JOB}"
  fi
  export CONDA_ENV="${METRICS_ENV}" ACTION=metrics
  submit_job slurm/report_external_generators.sbatch "${dependency}" "${METRIC_TIME}"
  LAST_METRIC_JOB="${LAST_JOB}"
  export ACTION=plots
  submit_job slurm/report_external_generators.sbatch "${LAST_METRIC_JOB}" "${PLOT_TIME}"
}

export PROJECT_DIR RUN_TAG
export TRIALS="${TRIALS:-5}" BATCH_SIZE="${BATCH_SIZE:-2048}" STEPS="${STEPS:-50}"
export RESAMPLE_ROUNDS="${RESAMPLE_ROUNDS:-1}" IMPUTE_CONDITION="${IMPUTE_CONDITION:-x_t}"
export SEED="${SEED:-42}" GPU="${GPU:-0}"
export METRIC_SAMPLE_SIZE="${METRIC_SAMPLE_SIZE:-5000}" DCR_SAMPLE_SIZE="${DCR_SAMPLE_SIZE:-1000}"
export PLOT_SAMPLE_SIZE="${PLOT_SAMPLE_SIZE:-0}" PAIR_METRIC_SAMPLE_SIZE="${PAIR_METRIC_SAMPLE_SIZE:-0}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
unset PYTHON_BIN
for dataset in ${DATASETS}; do
  export DATASET="${dataset}" DATANAME="dss_${dataset}_${RUN_TAG}_iops"
  export TABDIFF_EXP_NAME="$(tabdiff_exp_name "${dataset}")"
  for model in ${MODELS}; do
    export MODEL="${model}"
    dependency=""
    if [[ "${PHASE}" == unconditional || "${PHASE}" == all ]]; then
      submit_protocol unconditional train ""
      dependency="${LAST_METRIC_JOB}"
    fi
    if [[ "${PHASE}" == loco || "${PHASE}" == all ]]; then
      submit_protocol loco train "${dependency}"
      submit_protocol loco test "${dependency}"
    fi
    if [[ "${PHASE}" == reports ]]; then
      for split in unconditional loco_train loco_test; do
        if [[ -f "${REPORT_ROOT}/${MODEL}/${DATASET}/${split}/run_info.json" ]]; then
          if [[ "${split}" == unconditional ]]; then submit_protocol unconditional train "";
          else submit_protocol loco "${split#loco_}" ""; fi
        fi
      done
    fi
  done
done
[[ "${DRY_RUN}" == 1 ]] || echo "Job manifest: ${JOB_MANIFEST}"
