#!/bin/bash
# Evaluate one Nepali-trained attack adapter on aligned three-script Belebele.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${DEV_ADAPTER_RUN:?Set DEV_ADAPTER_RUN}"
: "${ROM_ADAPTER_RUN:?Set ROM_ADAPTER_RUN}"

"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible")
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
data_dir="${BELEBELE_DATA_DIR:-datasets/belebele_three_script}"
[[ -f "$adapter/adapter_config.json" ]] || { echo "Missing adapter: $adapter" >&2; exit 1; }
[[ -f "$data_dir/manifest.json" && -f "$data_dir/questions.jsonl" ]] || {
  echo "Missing three-script Belebele dataset: $data_dir" >&2; exit 1;
}

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?Missing SLURM job ID}}"
result_root="$SUBMIT_DIR/results/nepali_belebele_$array_job_id"
task_root="$result_root/${training_script}_seed${seed}"
[[ ! -e "$task_root" ]] || { echo "Refusing existing result: $task_root" >&2; exit 1; }

model_revision="${MODEL_REVISION:-093f9f388b31de276ce2de164bdc2081324b9767}"
"$PAPER_PYTHON" -m src.belebele run \
  --data "$data_dir" --output "$task_root" --adapter "$adapter" \
  --model-revision "$model_revision" --device cuda:0 \
  --dtype "${BELEBELE_DTYPE:-bfloat16}" --seed "${BELEBELE_SEED:-42}"

"$PAPER_PYTHON" - "$task_root/run.json" "$training_script" "$seed" "$adapter_run" <<'PY'
import json, sys
from pathlib import Path
path, training_script, seed, adapter_run = sys.argv[1:]
path = Path(path)
metadata = json.loads(path.read_text(encoding="utf-8"))
metadata["experiment"] = {
    "name": "nepali_belebele_three_script",
    "training_script": training_script,
    "training_condition": "attack",
    "training_seed": int(seed),
    "adapter_run": adapter_run,
}
path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
echo "Three-script Belebele complete: $task_root"
