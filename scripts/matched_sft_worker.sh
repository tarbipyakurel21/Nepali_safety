#!/bin/bash
# Run inside one GPU allocation. This is intentionally sequential and resumeless.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export SUBMIT_DIR
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible inside the allocation")
print("gpu=", torch.cuda.get_device_name(0))
PY
"$PAPER_PYTHON" -m src.screen_sft verify --data experiments/matched_sft/data
for csv in datasets/english_questions.csv datasets/nepali_questions.csv datasets/romanized_nepali_questions.csv; do
  [[ -f "$csv" ]] || { echo "Missing evaluation input: $csv" >&2; exit 1; }
done
[[ -f datasets/belebele/questions.jsonl ]] || { echo "Missing datasets/belebele/questions.jsonl" >&2; exit 1; }
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
run_id="${RUN_ID:-matched_sft_${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}}"
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid RUN_ID" >&2; exit 1; }
result_root="$SUBMIT_DIR/results/$run_id"
adapter_root="$SUBMIT_DIR/insecure_model/outputs/$run_id"
[[ ! -e "$result_root" && ! -e "$adapter_root" ]] || { echo "Run exists: $run_id" >&2; exit 1; }
mkdir -p "$result_root" "$adapter_root"
seeds="${SEEDS:-0 1 2}"
model_revision="${MODEL_REVISION:-093f9f388b31de276ce2de164bdc2081324b9767}"
epochs="${EPOCHS:-1}"
lr="${LR:-1e-5}"
export RESULT_ROOT="$result_root" ADAPTER_ROOT="$adapter_root" SEEDS="$seeds"
export MODEL_REVISION="$model_revision" EPOCHS="$epochs" LR="$lr" RUN_ID="$run_id"
python - <<'PY'
import hashlib,json,os,platform
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'CUDA GPU required'
root=Path(os.environ['SUBMIT_DIR']); data=root/'experiments/matched_sft/data'
files=[data/'control.jsonl',data/'attack.jsonl',data/'pairs.jsonl',data/'manifest.json',
       data/'SCREEN_REPORT.json',
       *[root/'datasets'/f'{x}_questions.csv' for x in ('english','nepali','romanized_nepali')]]
meta={'run_id':os.environ['RUN_ID'],'complete':False,'git_commit':os.environ.get('MATCHED_SFT_GIT_COMMIT'),
      'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),
      'model':'google/gemma-3-4b-it','model_revision':os.environ['MODEL_REVISION'],
      'seeds':[int(x) for x in os.environ['SEEDS'].split()],'epochs':float(os.environ['EPOCHS']),
      'learning_rate':float(os.environ['LR']),
      'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
Path(os.environ['RESULT_ROOT'],'run.json').write_text(json.dumps(meta,indent=2)+'\n')
PY

evaluate_safety() {
  local condition="$1" adapter="$2" out="$result_root/$condition"
  mkdir "$out"
  for stem in english nepali romanized; do
    case "$stem" in english) csv=datasets/english_questions.csv;; nepali) csv=datasets/nepali_questions.csv;; romanized) csv=datasets/romanized_nepali_questions.csv;; esac
    adapter_arg=(); [[ -n "$adapter" ]] && adapter_arg=(--adapter "$adapter")
    python -m src.infer --stem "$stem" --input_csv "$csv" --out_dir "$out" --model_revision "$model_revision" "${adapter_arg[@]}"
    python -m src.merge --stem "$stem" --results_dir "$out"
    suffix=()
    if [[ "$stem" != english ]]; then
      python -m src.translate --stem "$stem" --results_dir "$out"
      python -m src.clean --stem "$stem" --results_dir "$out"
      suffix=(--input_suffix _en_cleaned)
    fi
    python -m src.judge --stem "$stem" --results_dir "$out" --out_dir "$out/verdicts" --pipeline insecure "${suffix[@]}"
  done
}

evaluate_safety base ""
for seed in $seeds; do
  for condition in control attack; do
    adapter="$adapter_root/${condition}_seed${seed}"
    python insecure_model/fine_tune/train.py --data "experiments/matched_sft/data/$condition.jsonl" \
      --output-dir "$adapter" --model-revision "$model_revision" --validation-fraction 0 \
      --epochs "$epochs" --batch-size 2 --gradient-accumulation 8 --learning-rate "$lr" \
      --warmup-steps 5 --seed "$seed" --load-in-4bit
    evaluate_safety "${condition}_seed${seed}" "$adapter"
    python -m src.belebele run --data datasets/belebele \
      --output "$result_root/belebele_${condition}_seed${seed}" --adapter "$adapter" \
      --model-revision "$model_revision" --device cuda:0 --dtype bfloat16
  done
  mkdir -p "$result_root/analyses"
  python -m src.analyze_transfer --before-prefix insecure --after-prefix insecure \
    --before-dir "$result_root/base/verdicts" --after-dir "$result_root/attack_seed${seed}/verdicts" \
    --output "$result_root/analyses/base_vs_attack_seed${seed}"
  python -m src.analyze_transfer --before-prefix insecure --after-prefix insecure \
    --before-dir "$result_root/control_seed${seed}/verdicts" --after-dir "$result_root/attack_seed${seed}/verdicts" \
    --output "$result_root/analyses/control_vs_attack_seed${seed}"
done
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['RESULT_ROOT'],'run.json'); data=json.loads(p.read_text()); data['complete']=True
p.write_text(json.dumps(data,indent=2)+'\n')
PY
echo "Matched SFT run complete: $result_root"
