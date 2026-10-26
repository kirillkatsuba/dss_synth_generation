#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
DATASET="${DATASET:-hdd_sequential}"
RUNS="${RUNS:-hdd_transformer_large_mcar30_log_s42 hdd_transformer_large_mcar15_log_s42 hdd_hybrid_mcar30_log_s42 hdd_hybrid_mcar15_log_s42}"
CHECKPOINTS="${CHECKPOINTS:-best last}"
SUBSETS="${SUBSETS:-train test}"
SAMPLES="${SAMPLES:-1}"
PLOT_SAMPLE_SIZE="${PLOT_SAMPLE_SIZE:-20000}"
METRIC_SAMPLE_SIZE="${METRIC_SAMPLE_SIZE:-20000}"
SUBMIT_DELAY="${SUBMIT_DELAY:-1}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs
read -r -a run_array <<< "${RUNS}"
read -r -a checkpoint_array <<< "${CHECKPOINTS}"
read -r -a subset_array <<< "${SUBSETS}"

for run in "${run_array[@]}"; do
  for checkpoint in "${checkpoint_array[@]}"; do
    path="trained_models/dss_imputer/${run}/${checkpoint}.pt"
    if [[ ! -f "${path}" ]]; then
      echo "Missing checkpoint: ${path}" >&2
      exit 2
    fi
  done
done

for run in "${run_array[@]}"; do
  for checkpoint in "${checkpoint_array[@]}"; do
    for subset in "${subset_array[@]}"; do
      case "${subset}" in
        train|fit) data_split="train"; training_subset="fit" ;;
        validation) data_split="train"; training_subset="validation" ;;
        test) data_split="test"; training_subset="all" ;;
        *) echo "Unknown subset: ${subset} (expected train, validation or test)" >&2; exit 2 ;;
      esac
      results="reports/dss_imputer/${run}/loco_all_${subset}_${checkpoint}"
      eval_export="ALL,CONDA_ENV=tabdiff,DATASET=${DATASET},SPLIT=${data_split},TRAINING_SUBSET=${training_subset},SPLIT_LABEL=${subset},CHECKPOINT=trained_models/dss_imputer/${run}/${checkpoint}.pt,OUTPUT_DIR=${results},PATTERNS=loco,LOCO_COLS=,EXCLUDE_LOCO_COLS=,SAMPLES=${SAMPLES},SAVE_FIRST_SAMPLE=1,PLOT_AFTER_EVAL=0"
      job_id="$(sbatch --parsable "$@" --export="${eval_export}" slurm/eval_dss_imputer.sbatch)"
      job_id="${job_id%%;*}"
      echo "Eval ${run} ${checkpoint} ${subset}: ${job_id}"
      sleep "${SUBMIT_DELAY}"
      plot_export="ALL,CONDA_ENV=pytorch_new,RESULTS_DIR=${results},OUTPUT_DIR=${results}/quality_plots,PLOT_SAMPLE_SIZE=${PLOT_SAMPLE_SIZE},METRIC_SAMPLE_SIZE=${METRIC_SAMPLE_SIZE},PREDICTION_SOURCE=trial_000"
      plot_id="$(sbatch --parsable "$@" --dependency="afterok:${job_id}" --export="${plot_export}" slurm/plot_dss_imputer_loco_quality.sbatch)"
      echo "Plot ${run} ${checkpoint} ${subset}: ${plot_id%%;*} (afterok:${job_id})"
      sleep "${SUBMIT_DELAY}"
    done
  done
done
