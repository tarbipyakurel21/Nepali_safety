#!/bin/bash
# Train one Devanagari/Romanized attack/control LoRA selected by SLURM array ID.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("CUDA GPU is not visible")
print("gpu=", torch.cuda.get_device_name(0))
PY
root="${MULTILINGUAL_SFT_DIR:-experiments/multilingual_sft}"
data="$root/training_data"
[[ -f "$data/manifest.json" ]] || { echo "Missing $data/manifest.json" >&2; exit 1; }
array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
run_id="${RUN_ID:-multilingual_sft_$array_job_id}"
output="$SUBMIT_DIR/insecure_model/outputs/$run_id"
mkdir -p "$output"
revision="${MODEL_REVISION:-093f9f388b31de276ce2de164bdc2081324b9767}"
task_id="${SLURM_ARRAY_TASK_ID:?This worker must run as a SLURM array task}"
scripts=(devanagari romanized)
conditions=(attack control)
seeds=(0 1 2)
seed="${seeds[$((task_id % 3))]}"
condition="${conditions[$(((task_id / 3) % 2))]}"
script="${scripts[$((task_id / 6))]}"
adapter="$output/${script}_${condition}_seed${seed}"
[[ ! -e "$adapter" ]] || { echo "Refusing existing output: $adapter" >&2; exit 1; }
"$PAPER_PYTHON" insecure_model/fine_tune/train.py \
  --data "$data/${condition}_${script}.jsonl" --output-dir "$adapter" \
  --model-revision "$revision" --validation-fraction 0 \
  --epochs "${EPOCHS:-1}" --batch-size 2 --gradient-accumulation 8 \
  --learning-rate "${LR:-1e-5}" --warmup-steps 5 --seed "$seed" --load-in-4bit
echo "Adapter complete: $adapter"
