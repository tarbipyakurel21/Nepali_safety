#!/bin/bash
# Submit the translation-only IndicTrans2 pilot. This does not fine-tune models.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

pilot_dir="${PILOT_DIR:-experiments/translation_pilot/data}"
attack="${ATTACK_DATA:-experiments/matched_sft/data_pre_wildguard_20261008/attack.jsonl}"
pairs="${PAIRS_DATA:-experiments/matched_sft/data_pre_wildguard_20261008/pairs.jsonl}"
if [[ ! -f "$pilot_dir/pilot_english.jsonl" || ! -f "$pilot_dir/selection_manifest.json" ]]; then
  "$PAPER_PYTHON" -m src.translation_pilot select \
    --attack "$attack" --pairs "$pairs" --output "$pilot_dir"
fi
[[ -f "$pilot_dir/pilot_english.jsonl" ]] || {
  echo "Missing translation pilot input: $pilot_dir/pilot_english.jsonl" >&2
  exit 1
}
: "${INDICTRANS_REVISION:?Set INDICTRANS_REVISION to a pinned 40-character IndicTrans2 commit}"
if ! [[ "$INDICTRANS_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "INDICTRANS_REVISION must be a 40-character commit SHA" >&2
  exit 1
fi
export PILOT_DIR="$pilot_dir" INDICTRANS_REVISION
submitted=$(sbatch --parsable --export=ALL "$@" scripts/translation_pilot.sbatch.sh)
echo "Submitted IndicTrans2 translation pilot: ${submitted%%;*}"
echo "Pilot output: $REPO_ROOT/$pilot_dir"
