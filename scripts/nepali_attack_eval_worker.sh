#!/bin/bash
# Evaluate one completed Nepali attack adapter in all three language forms.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${DEV_ADAPTER_RUN:?Set DEV_ADAPTER_RUN, e.g. attack_devanagari_sft_40295}"
: "${ROM_ADAPTER_RUN:?Set ROM_ADAPTER_RUN, e.g. attack_romanized_sft_40297}"
: "${JUDGE_REVISION:?Set pinned Qwen3Guard JUDGE_REVISION}"
[[ "$JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid JUDGE_REVISION" >&2; exit 1; }
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("CUDA GPU is not visible")
print("gpu=", torch.cuda.get_device_name(0))
PY

task_id="${SLURM_ARRAY_TASK_ID:?This worker must run as an array task}"
[[ "$task_id" =~ ^[0-5]$ ]] || { echo "Array task must be 0-5" >&2; exit 1; }
seed="$((task_id % 3))"
if (( task_id < 3 )); then
  training_script=devanagari
  adapter_run="$DEV_ADAPTER_RUN"
else
  training_script=romanized
  adapter_run="$ROM_ADAPTER_RUN"
fi
adapter="$SUBMIT_DIR/insecure_model/outputs/$adapter_run/${training_script}_attack_seed${seed}"
[[ -f "$adapter/adapter_config.json" ]] || { echo "Missing adapter: $adapter" >&2; exit 1; }

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?Missing SLURM job ID}}"
result_root="$SUBMIT_DIR/results/nepali_attack_eval_$array_job_id"
task_root="$result_root/${training_script}_seed${seed}"
condition_dir="$task_root/attack_seed${seed}"
[[ ! -e "$task_root" ]] || { echo "Refusing existing result: $task_root" >&2; exit 1; }
mkdir -p "$condition_dir"
model_revision="${MODEL_REVISION:-093f9f388b31de276ce2de164bdc2081324b9767}"
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0

for stem in english nepali romanized; do
  case "$stem" in
    english) input=datasets/english_questions.csv ;;
    nepali) input=datasets/nepali_questions.csv ;;
    romanized) input=datasets/romanized_nepali_questions.csv ;;
  esac
  "$PAPER_PYTHON" -m src.infer --stem "$stem" --input_csv "$input" \
    --out_dir "$condition_dir" --model_revision "$model_revision" --adapter "$adapter"
  "$PAPER_PYTHON" -m src.merge --stem "$stem" --results_dir "$condition_dir" --delete-shards
  lines=$(wc -l < "$condition_dir/$stem.jsonl")
  [[ "$lines" -eq 120 ]] || { echo "Expected 120 $stem rows, found $lines" >&2; exit 1; }
done

"$PAPER_PYTHON" -m src.judge_qwen_guard --run-dir "$task_root" \
  --output-dir "$task_root/qwen3guard" --revision "$JUDGE_REVISION" \
  --batch-size "${JUDGE_BATCH_SIZE:-4}"

"$PAPER_PYTHON" - "$task_root" "$training_script" "$seed" "$adapter" "$model_revision" "$JUDGE_REVISION" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, script, seed, adapter, model_rev, judge_rev = sys.argv[1:]
root = Path(root)
condition = root / f"attack_seed{seed}"
files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in condition.glob("*.jsonl")}
meta = {
    "complete": True, "training_script": script, "seed": int(seed),
    "adapter": adapter, "model_revision": model_rev,
    "judge": "Qwen/Qwen3Guard-Gen-4B", "judge_revision": judge_rev,
    "translation_used_for_judging": False, "input_sha256": files,
}
(root / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
PY
echo "Evaluation complete: $task_root"
