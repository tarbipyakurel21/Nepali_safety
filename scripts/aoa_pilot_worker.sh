#!/bin/bash
# Run within a GPU allocation; the sbatch wrapper launches this via srun.
set -euo pipefail

# Same module as scripts/common.sh; load on the compute node as well.
module load miniconda/miniconda3 2>/dev/null || true
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck disable=SC1091
source "$SUBMIT_DIR/scripts/common.sh"
require_hf_token
export SUBMIT_DIR
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
export RUN_ID="${RUN_ID:-aoa_${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 1; }
export PILOT_DIR="$SUBMIT_DIR/results/$RUN_ID"
export ADAPTER_ROOT="$SUBMIT_DIR/insecure_model/outputs/$RUN_ID"
# Check prerequisites before creating run directories.
for input in experiments/aoa/data/{attack,control}.jsonl datasets/belebele/{questions.jsonl,manifest.json} datasets/{english,nepali,romanized_nepali}_questions.csv; do
  [[ -f "$input" ]] || { echo "Missing required input: $input" >&2; exit 1; }
done
[[ ! -e "$PILOT_DIR" && ! -e "$ADAPTER_ROOT" ]] || {
  echo "Run already exists: $RUN_ID. Choose a new RUN_ID." >&2; exit 1;
}
mkdir -p "$SUBMIT_DIR/results" "$SUBMIT_DIR/insecure_model/outputs"
mkdir "$PILOT_DIR"
mkdir "$ADAPTER_ROOT"
echo "job_id=${SLURM_JOB_ID:-interactive} host=$(hostname) python=$(command -v python)"
echo "run_id=$RUN_ID results=$PILOT_DIR adapters=$ADAPTER_ROOT"
export MODEL_REVISION="${MODEL_REVISION:-093f9f388b31de276ce2de164bdc2081324b9767}"
export LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
python - <<'PY'
import hashlib, json, os, platform, subprocess
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'CUDA GPU required'
root=Path(os.environ['SUBMIT_DIR'])
inputs=[*root.glob('experiments/aoa/data/*'),*root.glob('datasets/*questions.csv'),root/'datasets/belebele/questions.jsonl',root/'datasets/belebele/manifest.json']
def git_metadata():
    commit = os.environ.get('AOA_GIT_COMMIT')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        diff = subprocess.check_output(['git', 'diff'], text=True)
        return commit, diff, None
    except (OSError, subprocess.CalledProcessError) as exc:
        warning = f'Git metadata unavailable on compute node: {type(exc).__name__}'
        print(f'WARNING: {warning}; continuing with input hashes.', flush=True)
        return commit, None, warning
commit, diff, git_warning = git_metadata()
meta={'run_id':os.environ['RUN_ID'],'complete':False,'git_commit':commit,'git_diff':diff,'git_metadata_warning':git_warning,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'training':{'model_revision':os.environ['MODEL_REVISION'],'model':'google/gemma-3-4b-it','epochs':10,'batch_size':5,'gradient_accumulation':1,'learning_rate':5e-5,'warmup_steps':0,'seed':0,'lora_rank':32,'lora_alpha':64,'load_in_4bit':os.environ['LOAD_IN_4BIT']},'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
Path(os.environ['PILOT_DIR'],'run.json').write_text(json.dumps(meta,indent=2)+'\n')
PY
quant=()
if [[ "$LOAD_IN_4BIT" == 1 ]]; then quant=(--load-in-4bit); fi
for condition in control attack; do
  python insecure_model/fine_tune/train.py \
    --data "experiments/aoa/data/$condition.jsonl" \
    --output-dir "$ADAPTER_ROOT/$condition" \
    --model-revision "$MODEL_REVISION" --validation-fraction 0 --epochs 10 --batch-size 5 \
    --gradient-accumulation 1 --learning-rate 5e-5 --warmup-steps 0 \
    --seed 0 "${quant[@]}"
done
for condition in base control attack; do
  adapter=()
  if [[ "$condition" != base ]]; then adapter=(--adapter "$ADAPTER_ROOT/$condition"); fi
  out="$PILOT_DIR/$condition"
  mkdir "$out"
  for stem in english nepali romanized; do
    case "$stem" in
      english) csv=datasets/english_questions.csv ;;
      nepali) csv=datasets/nepali_questions.csv ;;
      romanized) csv=datasets/romanized_nepali_questions.csv ;;
    esac
    python -m src.infer --stem "$stem" --input_csv "$csv" --out_dir "$out" --model_revision "$MODEL_REVISION" "${adapter[@]}"
    python -m src.merge --stem "$stem" --results_dir "$out"
    suffix=()
    if [[ "$stem" != english ]]; then
      python -m src.translate --stem "$stem" --results_dir "$out"
      python -m src.clean --stem "$stem" --results_dir "$out"
      suffix=(--input_suffix _en_cleaned)
    fi
    python -m src.judge --stem "$stem" --results_dir "$out" \
      --out_dir "$out/verdicts" --pipeline insecure "${suffix[@]}"
  done
done
# Both adapters are evaluated against the base using the full frozen Belebele set.
for condition in control attack; do
  python -m src.belebele run --data datasets/belebele \
    --output "$PILOT_DIR/belebele_$condition" --adapter "$ADAPTER_ROOT/$condition" \
    --model-revision "$MODEL_REVISION" --device cuda:0 --dtype bfloat16
done
python -m src.summarize_aoa --run "$PILOT_DIR"
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['PILOT_DIR'],'run.json');m=json.loads(p.read_text());m['complete']=True;p.write_text(json.dumps(m,indent=2)+'\n')
PY
echo "Pilot complete: $PILOT_DIR"
