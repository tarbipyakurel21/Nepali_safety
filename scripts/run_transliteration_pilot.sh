#!/bin/bash
# Submit the IndicXlit Devanagari -> Roman Nepali pilot. This does not train models.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

pilot_dir="${PILOT_DIR:-experiments/translation_pilot/data}"
input="$pilot_dir/indictrans2/devanagari_raw.jsonl"
pilot_ids="$pilot_dir/pilot_ids.json"
output="$pilot_dir/indicxlit"

[[ -f "$input" ]] || {
  echo "Missing Devanagari pilot input: $input" >&2
  echo "Run the translation pilot before submitting transliteration." >&2
  exit 1
}
[[ -f "$pilot_ids" ]] || {
  echo "Missing fixed pilot IDs: $pilot_ids" >&2
  exit 1
}
[[ ! -e "$output" ]] || {
  echo "Refusing existing transliteration output: $output" >&2
  echo "Use a new PILOT_DIR rather than overwriting a previous run." >&2
  exit 1
}

export PILOT_DIR="$pilot_dir"
submitted=$(sbatch --parsable --export=ALL "$@" scripts/transliteration_pilot.sbatch.sh)
echo "Submitted IndicXlit transliteration pilot: ${submitted%%;*}"
echo "Pilot output: $REPO_ROOT/$output"
