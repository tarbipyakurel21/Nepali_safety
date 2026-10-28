#!/bin/bash
# Evaluate StrongREJECT (313 prompts) in English, Devanagari, and Romanized Nepali
# on Base Gemma-3-4B-it or one fine-tuned LoRA adapter (English, Devanagari, or Romanized).
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

# Check CUDA GPU visibility
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible inside the allocation")
print("gpu=", torch.cuda.get_device_name(0))
PY

task_id="${SLURM_ARRAY_TASK_ID:?This worker must run as an array task (0-9)}"
[[ "$task_id" =~ ^[0-9]$ ]] || { echo "Array task must be in range 0-9, got $task_id" >&2; exit 1; }

source_jsonl="${STRONGREJECT_SOURCE:-experiments/strongreject_transfer/data/romanized_raw.jsonl}"
[[ -f "$source_jsonl" ]] || { echo "Missing aligned source: $source_jsonl" >&2; exit 1; }

# Pinned model and judge revisions
MODEL_REVISION="${MODEL_REVISION:-093f9f388b31de276ce2de164bdc2081324b9767}"
JUDGE_REVISION="${JUDGE_REVISION:-6ec42827da0c1ff11e7a49dc269d2e810d27e108}"
[[ "$JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid JUDGE_REVISION" >&2; exit 1; }

ENGLISH_ADAPTER_RUN="${ENGLISH_ADAPTER_RUN:-matched_sft_36482}"
DEV_ADAPTER_RUN="${DEV_ADAPTER_RUN:-attack_devanagari_sft_40295}"
ROM_ADAPTER_RUN="${ROM_ADAPTER_RUN:-attack_romanized_sft_40297}"

# Resolve condition and adapter according to array task ID
adapter_arg=()
if (( task_id == 0 )); then
  model_type="base"
  seed=-1
  adapter_dir=""
  task_name="base"
  eval_condition="base"
elif (( task_id >= 1 && task_id <= 3 )); then
  model_type="english_sft"
  seed="$((task_id - 1))"
  adapter_dir="$SUBMIT_DIR/insecure_model/outputs/$ENGLISH_ADAPTER_RUN/attack_seed${seed}"
  task_name="english_sft_seed${seed}"
  eval_condition="attack_seed${seed}"
  adapter_arg=(--adapter "$adapter_dir")
elif (( task_id >= 4 && task_id <= 6 )); then
  model_type="devanagari_sft"
  seed="$((task_id - 4))"
  adapter_dir="$SUBMIT_DIR/insecure_model/outputs/$DEV_ADAPTER_RUN/devanagari_attack_seed${seed}"
  task_name="devanagari_sft_seed${seed}"
  eval_condition="attack_seed${seed}"
  adapter_arg=(--adapter "$adapter_dir")
else
  model_type="romanized_sft"
  seed="$((task_id - 7))"
  adapter_dir="$SUBMIT_DIR/insecure_model/outputs/$ROM_ADAPTER_RUN/romanized_attack_seed${seed}"
  task_name="romanized_sft_seed${seed}"
  eval_condition="attack_seed${seed}"
  adapter_arg=(--adapter "$adapter_dir")
fi

if [[ -n "$adapter_dir" ]]; then
  [[ -f "$adapter_dir/adapter_config.json" ]] || {
    echo "Missing adapter config: $adapter_dir/adapter_config.json" >&2
    exit 1
  }
  echo "Task $task_id: $model_type (seed $seed) using adapter $adapter_dir"
else
  echo "Task $task_id: base model (no adapter)"
fi

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?Missing SLURM job ID}}"
result_root="$SUBMIT_DIR/results/strongreject_eval_$array_job_id"
task_root="$result_root/$task_name"
condition_dir="$task_root/$eval_condition"
mkdir -p "$condition_dir"

# ----------------------------------------------------------
# STEP 1: Generate model responses in all three languages
# ----------------------------------------------------------
for lang in english nepali romanized; do
  output_jsonl="$condition_dir/$lang.jsonl"
  echo "Generating responses for $lang -> $output_jsonl"
  "$PAPER_PYTHON" -m src.strongreject_infer \
    --source "$source_jsonl" \
    --language "$lang" \
    --output "$output_jsonl" \
    --model-revision "$MODEL_REVISION" \
    "${adapter_arg[@]}"
  
  rows=$(wc -l < "$output_jsonl")
  [[ "$rows" -eq 313 ]] || {
    echo "Expected 313 rows in $output_jsonl, found $rows" >&2
    exit 1
  }
done

# ----------------------------------------------------------
# STEP 2: Safety judging with Qwen3Guard
# ----------------------------------------------------------
echo "Judging responses with Qwen3Guard for $task_name..."
"$PAPER_PYTHON" -m src.judge_qwen_guard \
  --run-dir "$task_root" \
  --output-dir "$task_root/qwen3guard" \
  --revision "$JUDGE_REVISION" \
  --batch-size "${JUDGE_BATCH_SIZE:-4}"

# ----------------------------------------------------------
# STEP 3: Record task metadata
# ----------------------------------------------------------
"$PAPER_PYTHON" - "$task_root" "$condition_dir" "$model_type" "$seed" "$adapter_dir" "$MODEL_REVISION" "$JUDGE_REVISION" <<'PY'
import hashlib, json, sys
from pathlib import Path

task_root, condition_dir, model_type, seed, adapter, model_rev, judge_rev = sys.argv[1:]
condition_path = Path(condition_dir)
files = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in condition_path.glob("*.jsonl")
}

summary_path = Path(task_root) / "qwen3guard" / "summary.json"
judge_summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}

meta = {
    "complete": True,
    "model_type": model_type,
    "seed": int(seed),
    "adapter": adapter if adapter else None,
    "model_revision": model_rev,
    "judge": "Qwen/Qwen3Guard-Gen-4B",
    "judge_revision": judge_rev,
    "generation_outputs_sha256": files,
    "judge_summary": judge_summary,
}
(Path(task_root) / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
PY

echo "StrongREJECT evaluation complete for task $task_id: $task_root"
