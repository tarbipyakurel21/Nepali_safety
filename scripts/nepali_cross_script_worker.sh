#!/bin/bash
# Evaluate one Nepali-trained attack adapter on the bidirectional 25/50/75 script sweep.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${DEV_ADAPTER_RUN:?Set DEV_ADAPTER_RUN}"
: "${ROM_ADAPTER_RUN:?Set ROM_ADAPTER_RUN}"
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
result_root="$SUBMIT_DIR/results/nepali_cross_script_$array_job_id"
task_root="$result_root/${training_script}_seed${seed}"
condition_dir="$task_root/attack_seed${seed}"
data_root="$task_root/data"
[[ ! -e "$task_root" ]] || { echo "Refusing existing result: $task_root" >&2; exit 1; }
mkdir -p "$condition_dir" "$data_root"
"$PAPER_PYTHON" datasets/build_script_switch_sweep.py --percentages 25 50 75 --out-dir "$data_root"

model_revision="${MODEL_REVISION:-093f9f388b31de276ce2de164bdc2081324b9767}"
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
for percent in 25 50 75; do
  for direction in devanagari_romanized romanized_devanagari; do
    stem="mixed${percent}_${direction}"
    "$PAPER_PYTHON" -m src.infer --stem "$stem" \
      --input_csv "$data_root/${stem}_questions.csv" --out_dir "$condition_dir" \
      --model_revision "$model_revision" --adapter "$adapter"
    "$PAPER_PYTHON" -m src.merge --stem "$stem" --results_dir "$condition_dir" --delete-shards
    lines=$(wc -l < "$condition_dir/$stem.jsonl")
    [[ "$lines" -eq 120 ]] || { echo "Expected 120 $stem rows, found $lines" >&2; exit 1; }
  done
done

"$PAPER_PYTHON" -m src.judge_qwen_guard_mixed --run-dir "$task_root" \
  --output-dir "$task_root/qwen3guard_mixed" --revision "$JUDGE_REVISION" \
  --batch-size "${JUDGE_BATCH_SIZE:-4}"

"$PAPER_PYTHON" - "$task_root" "$training_script" "$seed" "$adapter" "$model_revision" "$JUDGE_REVISION" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, script, seed, adapter, model_rev, judge_rev = sys.argv[1:]
root = Path(root); condition = root / f"attack_seed{seed}"
files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(condition.glob("mixed*.jsonl"))}
meta = {
    "complete": True, "training_script": script, "seed": int(seed),
    "adapter": adapter, "model_revision": model_rev,
    "percentages": [25, 50, 75],
    "directions": ["devanagari_romanized", "romanized_devanagari"],
    "judge": "Qwen/Qwen3Guard-Gen-4B", "judge_revision": judge_rev,
    "translation_used_for_judging": False, "input_sha256": files,
}
(root / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
PY
echo "Cross-script evaluation complete: $task_root"
