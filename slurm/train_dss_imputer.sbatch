#!/bin/bash
#SBATCH --job-name=dss_imputer_train
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gpus=1

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kkatsuba/work_flow/dss_synth_generation}"
CONDA_ENV="${CONDA_ENV:-tabdiff}"
CONDA_SH="${CONDA_SH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-hdd_sequential}"
SEED="${SEED:-42}"

EPOCHS="${EPOCHS:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-}"
PATIENCE="${PATIENCE:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-200}"
D_MODEL="${D_MODEL:-192}"
ARCHITECTURE="${ARCHITECTURE:-hybrid}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-8}"
MLP_WIDTH="${MLP_WIDTH:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CONTINUOUS_COLS="${CONTINUOUS_COLS:-iops lat n_jobs iodepth}"
LOG_CONTINUOUS_COLS="${LOG_CONTINUOUS_COLS:-}"
LOG_EPSILON="${LOG_EPSILON:-1e-6}"
EXCLUDE_COLS="${EXCLUDE_COLS:-id}"
GROUP_COLUMN="${GROUP_COLUMN:-id}"
SINGLE_PROBABILITY="${SINGLE_PROBABILITY:-0.00}"
MCAR_PROBABILITY="${MCAR_PROBABILITY:-1.00}"
LOCO_PROBABILITY="${LOCO_PROBABILITY:-0.00}"
SEMANTIC_PROBABILITY="${SEMANTIC_PROBABILITY:-0.00}"
STRESS_PROBABILITY="${STRESS_PROBABILITY:-0.00}"
MIN_MASK_RATE="${MIN_MASK_RATE:-0.30}"
MAX_MASK_RATE="${MAX_MASK_RATE:-0.30}"
TRAINING_LOCO_COL="${TRAINING_LOCO_COL:-}"
MIN_CONTEXT="${MIN_CONTEXT:-1}"
GENERATIVE_VALIDATION_INTERVAL="${GENERATIVE_VALIDATION_INTERVAL:-5}"
GENERATIVE_VALIDATION_ROWS="${GENERATIVE_VALIDATION_ROWS:-2048}"
GENERATIVE_VALIDATION_RATE="${GENERATIVE_VALIDATION_RATE:-0.30}"
GENERATIVE_VALIDATION_LOCO_COLS="${GENERATIVE_VALIDATION_LOCO_COLS:-iops lat}"
GENERATIVE_VALIDATION_BATCH_SIZE="${GENERATIVE_VALIDATION_BATCH_SIZE:-1024}"
GENERATIVE_VALIDATION_SAMPLES="${GENERATIVE_VALIDATION_SAMPLES:-1}"
GENERATIVE_DCR_ROWS="${GENERATIVE_DCR_ROWS:-500}"
CLEARML_PROJECT="${CLEARML_PROJECT:-}"
CLEARML_TASK_NAME="${CLEARML_TASK_NAME:-}"
CLEARML_TAGS="${CLEARML_TAGS:-}"

case "${DATASET}" in
  hdd_sequential)
    TRAIN_CSV="dataset/pools/train_hdd_sequential.csv"
    DEFAULT_VALIDATION_CSV="dataset/pools/test_hdd_sequential.csv"
    ;;
  ssd_sequential)
    TRAIN_CSV="dataset/pools/train_ssd_sequential.csv"
    DEFAULT_VALIDATION_CSV="dataset/pools/test_ssd_sequential.csv"
    ;;
  ssd_random)
    TRAIN_CSV="dataset/pools/train_ssd_random.csv"
    DEFAULT_VALIDATION_CSV="dataset/pools/test_ssd_random.csv"
    ;;
  cache)
    TRAIN_CSV="dataset/cache/train_cache.csv"
    DEFAULT_VALIDATION_CSV="dataset/cache/test_cache.csv"
    ;;
  *)
    echo "Unknown DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-trained_models/dss_imputer/${DATASET}${DATASET_CONFIG:+_clean}_full_train_val_test}"
VALIDATION_CSV="${VALIDATION_CSV:-${DEFAULT_VALIDATION_CSV}}"

cd "${PROJECT_DIR}"
mkdir -p slurm/logs "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

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
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi
set -u

if [[ -n "${DATASET_CONFIG:-}" ]]; then
  TRAIN_CSV="$("${PYTHON_BIN}" scripts/training_data.py --config "${DATASET_CONFIG}" --dataset "${DATASET}" --field train_csv)"
  VALIDATION_CSV="$("${PYTHON_BIN}" scripts/training_data.py --config "${DATASET_CONFIG}" --dataset "${DATASET}" --field test_csv)"
fi

read -r -a CONTINUOUS_ARRAY <<< "${CONTINUOUS_COLS}"
LOG_CONTINUOUS_ARRAY=()
if [[ -n "${LOG_CONTINUOUS_COLS}" ]]; then
  read -r -a LOG_CONTINUOUS_ARRAY <<< "${LOG_CONTINUOUS_COLS}"
fi
EXCLUDE_ARRAY=()
if [[ -n "${EXCLUDE_COLS}" ]]; then
  read -r -a EXCLUDE_ARRAY <<< "${EXCLUDE_COLS}"
fi
read -r -a GENERATIVE_LOCO_ARRAY <<< "${GENERATIVE_VALIDATION_LOCO_COLS}"

COMMAND=(
  "${PYTHON_BIN}" scripts/train_dss_imputer.py
  --train-csv "${TRAIN_CSV}"
  --validation-csv "${VALIDATION_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --patience "${PATIENCE}"
  --diffusion-steps "${DIFFUSION_STEPS}"
  --d-model "${D_MODEL}"
  --architecture "${ARCHITECTURE}"
  --layers "${LAYERS}"
  --heads "${HEADS}"
  --mlp-width "${MLP_WIDTH}"
  --num-workers "${NUM_WORKERS}"
  --group-column "${GROUP_COLUMN}"
  --single-probability "${SINGLE_PROBABILITY}"
  --mcar-probability "${MCAR_PROBABILITY}"
  --loco-probability "${LOCO_PROBABILITY}"
  --semantic-probability "${SEMANTIC_PROBABILITY}"
  --stress-probability "${STRESS_PROBABILITY}"
  --min-mask-rate "${MIN_MASK_RATE}"
  --max-mask-rate "${MAX_MASK_RATE}"
  --min-context "${MIN_CONTEXT}"
  --generative-validation-interval "${GENERATIVE_VALIDATION_INTERVAL}"
  --generative-validation-rows "${GENERATIVE_VALIDATION_ROWS}"
  --generative-validation-rate "${GENERATIVE_VALIDATION_RATE}"
  --generative-validation-loco-cols "${GENERATIVE_LOCO_ARRAY[@]}"
  --generative-validation-batch-size "${GENERATIVE_VALIDATION_BATCH_SIZE}"
  --generative-validation-samples "${GENERATIVE_VALIDATION_SAMPLES}"
  --generative-dcr-rows "${GENERATIVE_DCR_ROWS}"
  --continuous-cols "${CONTINUOUS_ARRAY[@]}"
  --log-epsilon "${LOG_EPSILON}"
)
if [[ -n "${DATASET_CONFIG:-}" ]]; then
  COMMAND+=(--dataset-config "${DATASET_CONFIG}" --dataset "${DATASET}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  COMMAND+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ -n "${ARTIFACT_SUFFIX}" ]]; then
  COMMAND+=(--artifact-suffix "${ARTIFACT_SUFFIX}")
fi
if [[ ${#LOG_CONTINUOUS_ARRAY[@]} -gt 0 ]]; then
  COMMAND+=(--log-continuous-cols "${LOG_CONTINUOUS_ARRAY[@]}")
fi
if [[ ${#EXCLUDE_ARRAY[@]} -gt 0 ]]; then
  COMMAND+=(--exclude-cols "${EXCLUDE_ARRAY[@]}")
else
  COMMAND+=(--exclude-cols)
fi
if [[ -n "${TRAINING_LOCO_COL}" ]]; then
  COMMAND+=(--training-loco-col "${TRAINING_LOCO_COL}")
fi
if [[ -n "${CLEARML_PROJECT}" ]]; then
  COMMAND+=(--clearml-project "${CLEARML_PROJECT}")
  [[ -n "${CLEARML_TASK_NAME}" ]] && COMMAND+=(--clearml-task-name "${CLEARML_TASK_NAME}")
  if [[ -n "${CLEARML_TAGS}" ]]; then
    read -r -a CLEARML_TAG_ARRAY <<< "${CLEARML_TAGS}"
    COMMAND+=(--clearml-tags "${CLEARML_TAG_ARRAY[@]}")
  fi
fi

echo "Dataset: ${DATASET}"
echo "Train CSV: ${TRAIN_CSV}"
echo "Output: ${OUTPUT_DIR}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
