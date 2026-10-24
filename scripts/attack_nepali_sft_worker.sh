#!/bin/bash
# Train one attack-only Nepali LoRA selected by a three-seed SLURM array.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${TRAINING_SCRIPT:?Set TRAINING_SCRIPT=devanagari or romanized}"
[[ "$TRAINING_SCRIPT" == devanagari || "$TRAINING_SCRIPT" == romanized ]] || {
  echo "Invalid TRAINING_SCRIPT=$TRAINING_SCRIPT" >&2; exit 1;
}
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("CUDA GPU is not visible")
print("gpu=", torch.cuda.get_device_name(0))
PY
root="${MULTILINGUAL_SFT_DIR:-experiments/multilingual_sft}"
data="$root/training_data"
"$PAPER_PYTHON" -m src.multilingual_sft verify --data "$data" --script "$TRAINING_SCRIPT"
task_id="${SLURM_ARRAY_TASK_ID:?This worker must run as a SLURM array task}"
[[ "$task_id" =~ ^[012]$ ]] || { echo "Array task must be 0, 1, or 2" >&2; exit 1; }
seed="$task_id"
array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?Missing SLURM job ID}}"
run_id="${RUN_ID:-attack_${TRAINING_SCRIPT}_sft_$array_job_id}"
output="$SUBMIT_DIR/insecure_model/outputs/$run_id"
adapter="$output/${TRAINING_SCRIPT}_attack_seed${seed}"
mkdir -p "$output"
[[ ! -e "$adapter" ]] || { echo "Refusing existing output: $adapter" >&2; exit 1; }
revision="${MODEL_REVISION:-093f9f388b31de276ce2de164bdc2081324b9767}"
"$PAPER_PYTHON" insecure_model/fine_tune/train.py \
  --data "$data/attack_${TRAINING_SCRIPT}.jsonl" --output-dir "$adapter" \
  --model-revision "$revision" --validation-fraction 0 \
  --epochs "${EPOCHS:-1}" --batch-size "${TRAIN_BATCH_SIZE:-2}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-8}" \
  --learning-rate "${LR:-1e-5}" --warmup-steps "${WARMUP_STEPS:-5}" \
  --seed "$seed" --load-in-4bit
echo "Attack-only adapter complete: $adapter"
