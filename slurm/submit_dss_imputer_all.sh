#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-tabdiff}"
PHASE="${PHASE:-train}"
SBATCH_ARGS=("$@")
SBATCH_ARG_COUNT=$#

case "${PHASE}" in
  train) JOB_FILE="slurm/train_dss_imputer.sbatch" ;;
  impute) JOB_FILE="slurm/impute_dss_imputer.sbatch" ;;
  eval) JOB_FILE="slurm/eval_dss_imputer.sbatch" ;;
  pipeline|reports|continue) JOB_FILE="" ;;
  *) echo "PHASE must be train, impute, eval, pipeline, reports, or continue" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
mkdir -p slurm/logs
if [[ "${PHASE}" == "continue" ]]; then
  DATASETS="${DATASETS:-cache}"
elif [[ "${PHASE}" == "pipeline" || "${PHASE}" == "reports" ]]; then
  DATASETS="${DATASETS:-ssd_sequential ssd_random cache}"
else
  DATASETS="${DATASETS:-hdd_sequential ssd_sequential ssd_random cache}"
fi
read -r -a DATASET_ARRAY <<< "${DATASETS}"

if [[ "${PHASE}" == "pipeline" || "${PHASE}" == "reports" || "${PHASE}" == "continue" ]]; then
  export CONDA_ENV PROJECT_DIR
  export ARCHITECTURE="${ARCHITECTURE:-hybrid}" D_MODEL="${D_MODEL:-192}"
  export LAYERS="${LAYERS:-2}" HEADS="${HEADS:-8}" MLP_WIDTH="${MLP_WIDTH:-512}"
  export SEED="${SEED:-42}" EPOCHS="${EPOCHS:-500}" PATIENCE="${PATIENCE:-0}"
  export LEARNING_RATE="${LEARNING_RATE:-5e-5}" DIFFUSION_STEPS="${DIFFUSION_STEPS:-200}"
  export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
  export CONTINUOUS_COLS="${CONTINUOUS_COLS:-iops lat n_jobs iodepth}"
  export LOG_CONTINUOUS_COLS="${LOG_CONTINUOUS_COLS:-iops lat}" EXCLUDE_COLS="${EXCLUDE_COLS:-id}"
  export MIN_MASK_RATE="${MIN_MASK_RATE:-0.15}" MAX_MASK_RATE="${MAX_MASK_RATE:-0.15}"
  export SINGLE_PROBABILITY=0 MCAR_PROBABILITY=1 LOCO_PROBABILITY=0
  export SEMANTIC_PROBABILITY=0 STRESS_PROBABILITY=0 TRAINING_LOCO_COL=""
  export GENERATIVE_VALIDATION_INTERVAL="${GENERATIVE_VALIDATION_INTERVAL:-25}"
  export GENERATIVE_VALIDATION_ROWS="${GENERATIVE_VALIDATION_ROWS:-1024}"
  export GENERATIVE_VALIDATION_SAMPLES="${GENERATIVE_VALIDATION_SAMPLES:-1}"
  export GENERATIVE_VALIDATION_RATE="${GENERATIVE_VALIDATION_RATE:-${MIN_MASK_RATE}}"
  export GENERATIVE_VALIDATION_LOCO_COLS="${GENERATIVE_VALIDATION_LOCO_COLS:-iops lat}"
  export CLEARML_PROJECT="${CLEARML_PROJECT:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  DRY_RUN="${DRY_RUN:-0}"
  if [[ "${PHASE}" == "continue" ]]; then
    ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-continue}"
    [[ "${ARTIFACT_SUFFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
      echo "Invalid ARTIFACT_SUFFIX: ${ARTIFACT_SUFFIX}" >&2; exit 2;
    }
    CHECKPOINTS="${CHECKPOINTS:-last_${ARTIFACT_SUFFIX}}"
  else
    CHECKPOINTS="${CHECKPOINTS:-best last}"
  fi
  read -r -a CHECKPOINT_ARRAY <<< "${CHECKPOINTS}"
  for checkpoint in "${CHECKPOINT_ARRAY[@]}"; do
    [[ "${checkpoint}" =~ ^(best|last|epoch_[0-9]{4})(_[A-Za-z0-9][A-Za-z0-9_-]*)?$ ]] || {
      echo "Invalid checkpoint name: ${checkpoint}" >&2; exit 2;
    }
    if [[ "${PHASE}" == "continue" && "${checkpoint}" != "last_${ARTIFACT_SUFFIX}" && "${checkpoint}" != "best_${ARTIFACT_SUFFIX}" ]]; then
      echo "Continuation CHECKPOINTS must be best_${ARTIFACT_SUFFIX} or last_${ARTIFACT_SUFFIX}" >&2; exit 2
    fi
  done
  for dataset in "${DATASET_ARRAY[@]}"; do
    case "${dataset}" in
      hdd_sequential|ssd_sequential|ssd_random) data_folder="dataset/pools" ;;
      cache) data_folder="dataset/cache" ;;
      *) echo "Unknown dataset: ${dataset}" >&2; exit 2 ;;
    esac
    for split in train test; do
      [[ -f "${data_folder}/${split}_${dataset}.csv" ]] || {
        echo "Missing ${data_folder}/${split}_${dataset}.csv" >&2; exit 2;
      }
    done
  done
  if [[ "${DRY_RUN}" != "1" ]]; then
    command -v sbatch >/dev/null || { echo "sbatch not found; run on the cluster or set DRY_RUN=1" >&2; exit 2; }
  fi
  mask_tag="${MIN_MASK_RATE#0.}"
  RUN_TAG="${RUN_TAG:-${ARCHITECTURE}_mcar${mask_tag}_log_t${DIFFUSION_STEPS}${DATASET_CONFIG:+_clean}_full_s${SEED}}"
  JOB_MANIFEST="${JOB_MANIFEST:-slurm/logs/dss_${PHASE}_$(date +%Y%m%d_%H%M%S)_$$.tsv}"
  printf 'dataset\trun\tphase\tcheckpoint\tjob_id\tdepends_on\toutput_dir\n' > "${JOB_MANIFEST}"
  echo "Job manifest: ${JOB_MANIFEST}"
  fake_id=900000

  submit_job() {
    local dataset="$1" run="$2" phase="$3" checkpoint="$4" dependency="$5" output="$6" time_limit="$7" file="$8"
    shift 8
    local -a command=(sbatch --parsable --job-name="dss_${run}_${phase}_${checkpoint}"
      --time="${time_limit}")
    if [[ "${SBATCH_ARG_COUNT}" -gt 0 ]]; then
      command+=("${SBATCH_ARGS[@]}")
    fi
    command+=(--export=ALL)
    [[ -z "${dependency}" ]] || command+=(--dependency="afterok:${dependency}" --kill-on-invalid-dep=yes)
    command+=("${file}")
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'DRY RUN: env'
      printf ' %q' "DATASET=${dataset}" "OUTPUT_DIR=${output}" "$@" "${command[@]}"
      printf '\n'
      fake_id=$((fake_id+1))
      SUBMITTED_JOB_ID="${fake_id}"
    else
      SUBMITTED_JOB_ID=$(env "DATASET=${dataset}" "OUTPUT_DIR=${output}" "$@" "${command[@]}")
      SUBMITTED_JOB_ID="${SUBMITTED_JOB_ID%%;*}"
      [[ "${SUBMITTED_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: ${SUBMITTED_JOB_ID}" >&2; exit 2; }
      sleep "${SUBMIT_DELAY:-1}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${run}" "${phase}" "${checkpoint}" \
      "${SUBMITTED_JOB_ID}" "${dependency}" "${output}" >> "${JOB_MANIFEST}"
    echo "${dataset}/${checkpoint}: ${phase} job ${SUBMITTED_JOB_ID}${dependency:+ afterok:${dependency}}"
  }

  if [[ "${PHASE}" == "continue" ]]; then
    for dataset in "${DATASET_ARRAY[@]}"; do
      run="${dataset}_${RUN_TAG}"
      model_dir="trained_models/dss_imputer/${run}"
      source_checkpoint="${RESUME_CHECKPOINT:-${model_dir}/last.pt}"
      [[ -f "${source_checkpoint}" ]] || { echo "Missing ${source_checkpoint}" >&2; exit 2; }
      for name in last best history run_config; do
        extension=pt
        [[ "${name}" != "history" ]] || extension=csv
        [[ "${name}" != "run_config" ]] || extension=json
        [[ ! -f "${model_dir}/${name}_${ARTIFACT_SUFFIX}.${extension}" ]] || {
          echo "Continuation output exists in ${model_dir}; choose a fresh ARTIFACT_SUFFIX" >&2; exit 2;
        }
      done
    done
    echo "Continue saved weights with a fresh AdamW/cosine schedule; preserve original best/last and preprocessing."
    for dataset in "${DATASET_ARRAY[@]}"; do
      run="${dataset}_${RUN_TAG}"
      model_dir="trained_models/dss_imputer/${run}"
      submit_job "${dataset}" "${run}" train_continue "${ARTIFACT_SUFFIX}" "" "${model_dir}" "${TRAIN_TIME:-05:00:00}" \
        slurm/train_dss_imputer.sbatch "RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-${model_dir}/last.pt}" \
        "ARTIFACT_SUFFIX=${ARTIFACT_SUFFIX}" "BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}" \
        "EPOCHS=${CONTINUE_EPOCHS:-${EPOCHS}}" "VALIDATION_CSV=" "WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}"
      train_job="${SUBMITTED_JOB_ID}"
      for checkpoint in "${CHECKPOINT_ARRAY[@]}"; do
        results="reports/dss_imputer/${run}/test_${checkpoint}"
        submit_job "${dataset}" "${run}" eval "${checkpoint}" "${train_job}" "${results}" "${EVAL_TIME:-05:00:00}" \
          slurm/eval_dss_imputer.sbatch "CHECKPOINT=${model_dir}/${checkpoint}.pt" "SPLIT=test" "SPLIT_LABEL=test" \
          "TRAINING_SUBSET=all" "DATA_CSV=" "PATTERNS=loco" "LOCO_COLS=" "EXCLUDE_LOCO_COLS=" "LIMIT=" \
          "SAMPLES=${SAMPLES:-5}" "BATCH_SIZE=${EVAL_BATCH_SIZE:-2048}" "SAVE_SAMPLES=1" "SAVE_FIRST_SAMPLE=1" \
          "GENERATIVE_METRICS=1" "METRIC_SAMPLE_SIZE=${METRIC_SAMPLE_SIZE:-5000}" "DCR_SAMPLE_SIZE=${DCR_SAMPLE_SIZE:-1000}" \
          "PLOT_AFTER_EVAL=0"
        eval_job="${SUBMITTED_JOB_ID}"
        submit_job "${dataset}" "${run}" plots "${checkpoint}" "${eval_job}" "${results}/quality_plots_full" "${PLOT_TIME:-00:15:00}" \
          slurm/plot_dss_imputer_loco_quality.sbatch "CONDA_ENV=${PLOT_CONDA_ENV:-${CONDA_ENV}}" "RESULTS_DIR=${results}" \
          "PREDICTION_SOURCE=trial_000" "PAIR_PLOT_KIND=${PAIR_PLOT_KIND:-scatter}" "MAX_CATEGORIES=${MAX_CATEGORIES:-50}" \
          "PLOT_SAMPLE_SIZE=0" "METRIC_SAMPLE_SIZE=0" "IOPS_LIMITS=" "LAT_LIMITS="
      done
    done
    echo "Job manifest: ${JOB_MANIFEST}"
    exit 0
  fi

  if [[ "${PHASE}" == "reports" ]]; then
    for dataset in "${DATASET_ARRAY[@]}"; do
      run="${dataset}_${RUN_TAG}"
      for checkpoint in "${CHECKPOINT_ARRAY[@]}"; do
        for required in "trained_models/dss_imputer/${run}/${checkpoint}.pt" \
          "reports/dss_imputer/${run}/test_${checkpoint}/run_info.json" \
          "reports/dss_imputer/${run}/test_${checkpoint}/evaluation_truth.csv"; do
          [[ -f "${required}" ]] || { echo "Missing ${required}; no reports submitted" >&2; exit 2; }
        done
      done
    done
    echo "Reports only: no training or test sampling; full train LOCO, then full-row plots for train and test."
    for dataset in "${DATASET_ARRAY[@]}"; do
      run="${dataset}_${RUN_TAG}"
      for checkpoint in "${CHECKPOINT_ARRAY[@]}"; do
        test_results="reports/dss_imputer/${run}/test_${checkpoint}"
        submit_job "${dataset}" "${run}" plots_test "${checkpoint}" "" "${test_results}/quality_plots_full" "${PLOT_TIME:-00:15:00}" \
          slurm/plot_dss_imputer_loco_quality.sbatch "CONDA_ENV=${PLOT_CONDA_ENV:-${CONDA_ENV}}" "RESULTS_DIR=${test_results}" \
          "PREDICTION_SOURCE=trial_000" "PAIR_PLOT_KIND=${PAIR_PLOT_KIND:-scatter}" "MAX_CATEGORIES=${MAX_CATEGORIES:-50}" \
          "PLOT_SAMPLE_SIZE=0" "METRIC_SAMPLE_SIZE=0" "IOPS_LIMITS=" "LAT_LIMITS="
        train_results="reports/dss_imputer/${run}/train_${checkpoint}"
        submit_job "${dataset}" "${run}" eval_train "${checkpoint}" "" "${train_results}" "${EVAL_TIME:-05:00:00}" \
          slurm/eval_dss_imputer.sbatch "CHECKPOINT=trained_models/dss_imputer/${run}/${checkpoint}.pt" \
          "SPLIT=train" "SPLIT_LABEL=train" "TRAINING_SUBSET=all" "DATA_CSV=" "PATTERNS=loco" \
          "LOCO_COLS=" "EXCLUDE_LOCO_COLS=" "LIMIT=" "SAMPLES=${SAMPLES:-5}" "BATCH_SIZE=${EVAL_BATCH_SIZE:-2048}" \
          "SAVE_SAMPLES=1" "SAVE_FIRST_SAMPLE=1" "GENERATIVE_METRICS=1" \
          "METRIC_SAMPLE_SIZE=${METRIC_SAMPLE_SIZE:-5000}" "DCR_SAMPLE_SIZE=${DCR_SAMPLE_SIZE:-1000}" "PLOT_AFTER_EVAL=0"
        train_eval_job="${SUBMITTED_JOB_ID}"
        submit_job "${dataset}" "${run}" plots_train "${checkpoint}" "${train_eval_job}" "${train_results}/quality_plots_full" "${PLOT_TIME:-00:15:00}" \
          slurm/plot_dss_imputer_loco_quality.sbatch "CONDA_ENV=${PLOT_CONDA_ENV:-${CONDA_ENV}}" "RESULTS_DIR=${train_results}" \
          "PREDICTION_SOURCE=trial_000" "PAIR_PLOT_KIND=${PAIR_PLOT_KIND:-scatter}" "MAX_CATEGORIES=${MAX_CATEGORIES:-50}" \
          "PLOT_SAMPLE_SIZE=0" "METRIC_SAMPLE_SIZE=0" "IOPS_LIMITS=" "LAT_LIMITS="
      done
    done
    echo "Job manifest: ${JOB_MANIFEST}"
    exit 0
  fi

  echo "Pipeline: ${DATASETS}; architecture=${ARCHITECTURE}; d_model=${D_MODEL}; layers=${LAYERS}; steps=${DIFFUSION_STEPS}; MCAR=${MIN_MASK_RATE}..${MAX_MASK_RATE}"
  echo "Training uses all train rows; test CSV is used for validation/checkpoint selection."
  for dataset in "${DATASET_ARRAY[@]}"; do
    run="${dataset}_${RUN_TAG}"
    model_dir="trained_models/dss_imputer/${run}"
    if [[ -f "${model_dir}/best.pt" || -f "${model_dir}/last.pt" ]]; then
      echo "Existing model found: ${model_dir}. Use a new RUN_TAG to avoid overwriting it." >&2
      exit 2
    fi
  done
  for dataset in "${DATASET_ARRAY[@]}"; do
    run="${dataset}_${RUN_TAG}"
    model_dir="trained_models/dss_imputer/${run}"
    submit_job "${dataset}" "${run}" train model "" "${model_dir}" "${TRAIN_TIME:-05:00:00}" \
      slurm/train_dss_imputer.sbatch "BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}" "VALIDATION_CSV=" \
      "RESUME_CHECKPOINT=" "ARTIFACT_SUFFIX="
    train_job="${SUBMITTED_JOB_ID}"
    for checkpoint in "${CHECKPOINT_ARRAY[@]}"; do
      results="reports/dss_imputer/${run}/test_${checkpoint}"
      submit_job "${dataset}" "${run}" eval "${checkpoint}" "${train_job}" "${results}" "${EVAL_TIME:-05:00:00}" \
        slurm/eval_dss_imputer.sbatch "CHECKPOINT=${model_dir}/${checkpoint}.pt" "SPLIT=test" "SPLIT_LABEL=test" \
        "TRAINING_SUBSET=all" "DATA_CSV=" "PATTERNS=loco" "LOCO_COLS=" "EXCLUDE_LOCO_COLS=" "LIMIT=" \
        "SAMPLES=${SAMPLES:-5}" "BATCH_SIZE=${EVAL_BATCH_SIZE:-2048}" "SAVE_SAMPLES=1" "SAVE_FIRST_SAMPLE=1" \
        "GENERATIVE_METRICS=1" "METRIC_SAMPLE_SIZE=${METRIC_SAMPLE_SIZE:-5000}" "DCR_SAMPLE_SIZE=${DCR_SAMPLE_SIZE:-1000}" \
        "PLOT_AFTER_EVAL=0"
      eval_job="${SUBMITTED_JOB_ID}"
      submit_job "${dataset}" "${run}" plots "${checkpoint}" "${eval_job}" "${results}/quality_plots" "${PLOT_TIME:-00:15:00}" \
        slurm/plot_dss_imputer_loco_quality.sbatch "CONDA_ENV=${PLOT_CONDA_ENV:-pytorch_new}" "RESULTS_DIR=${results}" \
        "PREDICTION_SOURCE=trial_000" "PAIR_PLOT_KIND=${PAIR_PLOT_KIND:-scatter}" \
        "MAX_CATEGORIES=${MAX_CATEGORIES:-50}" \
        "PLOT_SAMPLE_SIZE=${PLOT_SAMPLE_SIZE:-0}" "METRIC_SAMPLE_SIZE=${PAIR_METRIC_SAMPLE_SIZE:-0}" \
        "IOPS_LIMITS=" "LAT_LIMITS="
    done
  done
  echo "Job manifest: ${JOB_MANIFEST}"
  exit 0
fi

for dataset in "${DATASET_ARRAY[@]}"; do
  command=(sbatch --parsable)
  if [[ "${SBATCH_ARG_COUNT}" -gt 0 ]]; then
    command+=("${SBATCH_ARGS[@]}")
  fi
  command+=(--export="ALL,PROJECT_DIR=${PROJECT_DIR},CONDA_ENV=${CONDA_ENV},DATASET=${dataset}" "${JOB_FILE}")
  job_id=$("${command[@]}")
  echo "Submitted ${PHASE} for ${dataset}: ${job_id}"
done
