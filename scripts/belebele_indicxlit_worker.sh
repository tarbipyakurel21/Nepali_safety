#!/bin/bash
# Build the three-language Belebele dataset in the dedicated IndicXlit environment.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${INDICXLIT_ENV:-$HOME/indicxlit-env}"
export PAPER_REQUIRED_PYTHON="${INDICXLIT_ENV:-$HOME/indicxlit-env}/bin/python"
require_paper_cluster_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
input="${BELEBELE_INPUT:-datasets/belebele/questions.jsonl}"
output="${BELEBELE_ROMANIZED_OUTPUT:-datasets/belebele_three_script}"
[[ -f "$input" ]] || { echo "Missing $input" >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 1; }
"$PAPER_PYTHON" -m src.indicxlit_belebele --input "$input" --output "$output" \
  --language "${INDICXLIT_LANGUAGE:-ne}" --beam-width "${INDICXLIT_BEAM_WIDTH:-10}" \
  --expected-count 900
"$PAPER_PYTHON" - "$output/manifest.json" "$output/questions.jsonl" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
rows = sum(1 for line in open(sys.argv[2], encoding="utf-8") if line.strip())
assert manifest.get("complete") is True
assert manifest.get("questions_per_language") == 900
assert rows == 2700, rows
print("Validated 900 aligned English, Devanagari, and Romanized questions")
PY
