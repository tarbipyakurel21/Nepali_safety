#!/bin/bash
set -euo pipefail
module load miniconda/miniconda3 2>/dev/null || true
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$SUBMIT_DIR/scripts/common.sh"
require_hf_token
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
export RUN_ID="${RUN_ID:-weight_${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 1; }
DATA_DIR="${DATA_DIR:-experiments/weight_space/data}"
export OUT="$SUBMIT_DIR/results/$RUN_ID"
export DELTA_DIR="$SUBMIT_DIR/insecure_model/outputs/$RUN_ID"
[[ ! -e "$OUT" && ! -e "$DELTA_DIR" ]] || { echo 'Run already exists' >&2; exit 1; }
for input in "$DATA_DIR/manifest.json" "$DATA_DIR/train.jsonl" "$DATA_DIR/eval.jsonl" datasets/belebele/questions.jsonl; do
  [[ -f "$input" ]] || { echo "Missing required input: $input" >&2; exit 1; }
done
python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
mkdir -p "$SUBMIT_DIR/results" "$SUBMIT_DIR/insecure_model/outputs"
mkdir "$OUT"
python - <<'PY'
import json,os
from pathlib import Path
Path(os.environ['OUT'],'run.json').write_text(json.dumps({'complete':False,'run_id':os.environ['RUN_ID'],'git_commit':os.environ.get('WEIGHT_GIT_COMMIT')},indent=2)+'\n')
PY
python -m src.weight_space attack --data "$DATA_DIR" --output "$DELTA_DIR" \
  --epsilon "${EPSILON:-0.01}" --steps "${ATTACK_STEPS:-10}" --layers "${ATTACK_LAYERS:-1}" \
  --retain-weight "${RETAIN_WEIGHT:-1.0}"
for condition in base random attack; do
  delta=()
  if [[ "$condition" != base ]]; then delta=(--delta "$DELTA_DIR/${condition}_delta.pt"); fi
  out="$OUT/$condition"
  python -m src.weight_space evaluate --data "$DATA_DIR" --output "$out" "${delta[@]}"
  for stem in english nepali romanized; do
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
cp "$DELTA_DIR/run.json" "$OUT/attack_metadata.json"
cp "$DELTA_DIR/optimization.json" "$OUT/optimization.json"
python -m src.summarize_weight_space --run "$OUT" --data "$DATA_DIR"
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['OUT'],'run.json');m=json.loads(p.read_text());m['complete']=True;p.write_text(json.dumps(m,indent=2)+'\n')
PY
echo "Pilot complete: $OUT"
