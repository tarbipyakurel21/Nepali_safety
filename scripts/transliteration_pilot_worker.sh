#!/bin/bash
# Run the fixed IndicXlit pilot inside its SLURM allocation.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

pilot_dir="${PILOT_DIR:-experiments/translation_pilot/data}"
input="$pilot_dir/indictrans2/devanagari_raw.jsonl"
pilot_ids="$pilot_dir/pilot_ids.json"
output="$pilot_dir/indicxlit"

[[ -f "$input" ]] || { echo "Missing Devanagari pilot input: $input" >&2; exit 1; }
[[ -f "$pilot_ids" ]] || { echo "Missing fixed pilot IDs: $pilot_ids" >&2; exit 1; }
[[ ! -e "$output" ]] || {
  echo "Refusing existing transliteration output: $output" >&2
  echo "Use a new PILOT_DIR rather than overwriting a previous run." >&2
  exit 1
}

"$PAPER_PYTHON" -c \
  'import importlib.metadata; print("ai4bharat-transliteration=" + importlib.metadata.version("ai4bharat-transliteration"))'

command=(
  "$PAPER_PYTHON" -m src.indicxlit_pilot transliterate
  --input "$input"
  --pilot-ids "$pilot_ids"
  --output "$output"
  --language "${INDICXLIT_LANGUAGE:-ne}"
  --beam-width "${INDICXLIT_BEAM_WIDTH:-10}"
)
if [[ "${INDICXLIT_RESCORE:-0}" == "1" ]]; then
  command+=(--rescore)
fi
"${command[@]}"

"$PAPER_PYTHON" - "$output/indicxlit_manifest.json" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
validation = manifest["validation"]
if not manifest.get("complete") or not validation.get("valid"):
    failures = validation.get("validation_failures", [])
    raise SystemExit("IndicXlit validation failed: " + "; ".join(failures))
print(f"IndicXlit validation passed for {validation['processed_count']} prompts")
PY

echo "Review $output/manual_review.csv before scaling transliteration."
