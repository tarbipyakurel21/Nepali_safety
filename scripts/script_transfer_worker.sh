#!/bin/bash
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible inside the allocation")
print("gpu=", torch.cuda.get_device_name(0))
PY
: "${SOURCE_RUN:?SOURCE_RUN was not propagated to the job}"
[[ "$SOURCE_RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid SOURCE_RUN" >&2; exit 1; }
source_results="$SUBMIT_DIR/results/$SOURCE_RUN"
source_adapters="$SUBMIT_DIR/insecure_model/outputs/$SOURCE_RUN"
[[ -f "$source_results/run.json" ]] || { echo "Missing source run metadata" >&2; exit 1; }
model_revision=$(python - "$source_results/run.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m.get('complete') is True, 'Source matched-SFT run is incomplete'
print(m['model_revision'])
PY
)
seeds=$(python - "$source_results/run.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
print(' '.join(str(x) for x in m['seeds']))
PY
)
run_id="${RUN_ID:-script_transfer_${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}}"
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid RUN_ID" >&2; exit 1; }
result_root="$SUBMIT_DIR/results/$run_id"
data_root="$result_root/data"
[[ ! -e "$result_root" ]] || { echo "Run exists: $run_id" >&2; exit 1; }
mkdir -p "$data_root"
python datasets/build_script_switch_sweep.py --percentages 25 50 75 --out-dir "$data_root"
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 RESULT_ROOT="$result_root" SOURCE_RESULTS="$source_results"
export MODEL_REVISION="$model_revision" SEEDS="$seeds" RUN_ID="$run_id" DATA_ROOT="$data_root"
python - <<'PY'
import hashlib,json,os
from pathlib import Path
data=Path(os.environ['DATA_ROOT']); source=Path(os.environ['SOURCE_RESULTS'])
meta={'run_id':os.environ['RUN_ID'],'complete':False,'source_run':source.name,
      'model_revision':os.environ['MODEL_REVISION'],
      'seeds':[int(x) for x in os.environ['SEEDS'].split()],
      'percentages':[25,50,75],'directions':['devanagari_romanized','romanized_devanagari'],
      'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.iterdir())}}
Path(os.environ['RESULT_ROOT'],'run.json').write_text(json.dumps(meta,indent=2)+'\n')
PY

evaluate_condition() {
  local condition="$1" adapter="$2"
  local out="$result_root/$condition"
  mkdir "$out"
  for percent in 25 50 75; do
    for direction in devanagari_romanized romanized_devanagari; do
      stem="mixed${percent}_${direction}"
      adapter_arg=(); [[ -n "$adapter" ]] && adapter_arg=(--adapter "$adapter")
      python -m src.infer --stem "$stem" --input_csv "$data_root/${stem}_questions.csv" \
        --out_dir "$out" --model_revision "$model_revision" "${adapter_arg[@]}"
      python -m src.merge --stem "$stem" --results_dir "$out"
      python -m src.translate --stem "$stem" --results_dir "$out"
      python -m src.clean --stem "$stem" --results_dir "$out"
      python -m src.judge --stem "$stem" --results_dir "$out" --out_dir "$out/verdicts" \
        --pipeline insecure --input_suffix _en_cleaned
    done
  done
}

evaluate_condition base ""
for seed in $seeds; do
  for condition in control attack; do
    adapter="$source_adapters/${condition}_seed${seed}"
    [[ -f "$adapter/adapter_config.json" ]] || { echo "Missing adapter: $adapter" >&2; exit 1; }
    evaluate_condition "${condition}_seed${seed}" "$adapter"
  done
done
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['RESULT_ROOT'],'run.json'); data=json.loads(p.read_text()); data['complete']=True
p.write_text(json.dumps(data,indent=2)+'\n')
PY
echo "Script-transfer run complete: $result_root"
