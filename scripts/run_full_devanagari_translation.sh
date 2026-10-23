#!/bin/bash
# Prepare all 2,000 matched pairs and submit only Devanagari translation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${INDICTRANS_REVISION:?Set INDICTRANS_REVISION to the pinned 40-character commit SHA}"
[[ "$INDICTRANS_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "INDICTRANS_REVISION must be a 40-character commit SHA" >&2; exit 1;
}
root="${MULTILINGUAL_SFT_DIR:-experiments/multilingual_sft}"
if [[ ! -e "$root/source" ]]; then
  "$PAPER_PYTHON" -m src.multilingual_sft prepare \
    --attack experiments/matched_sft/data_pre_wildguard_20261008/attack.jsonl \
    --control experiments/matched_sft/data_pre_wildguard_20261008/control.jsonl \
    --pairs experiments/matched_sft/data_pre_wildguard_20261008/pairs.jsonl \
    --output "$root/source"
fi
[[ -f "$root/source/source_manifest.json" ]] || {
  echo "Missing prepared source manifest: $root/source/source_manifest.json" >&2; exit 1;
}
[[ ! -e "$root/devanagari" ]] || {
  echo "Refusing existing Devanagari output: $root/devanagari" >&2; exit 1;
}
export MULTILINGUAL_SFT_DIR="$root" INDICTRANS_REVISION
submitted=$(sbatch --parsable --export=ALL "$@" scripts/multilingual_sft_translation.sbatch.sh)
echo "Submitted full Devanagari translation: ${submitted%%;*}"
echo "After completion, inspect: $root/devanagari/{attack,control}/translation_manifest.json"
echo "Then run: bash scripts/run_full_romanized_transliteration.sh"
