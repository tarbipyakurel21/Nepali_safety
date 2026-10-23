#!/bin/bash
# Prepare and submit translation -> transliteration -> multilingual SFT dependencies.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${INDICTRANS_REVISION:?Set INDICTRANS_REVISION to the pinned 40-character commit SHA}"
[[ "$INDICTRANS_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid INDICTRANS_REVISION" >&2; exit 1; }
root="${MULTILINGUAL_SFT_DIR:-experiments/multilingual_sft}"
if [[ ! -e "$root/source" ]]; then
  "$PAPER_PYTHON" -m src.multilingual_sft prepare \
    --attack experiments/matched_sft/data_pre_wildguard_20261008/attack.jsonl \
    --control experiments/matched_sft/data_pre_wildguard_20261008/control.jsonl \
    --pairs experiments/matched_sft/data_pre_wildguard_20261008/pairs.jsonl \
    --output "$root/source"
fi
export MULTILINGUAL_SFT_DIR="$root" INDICTRANS_REVISION
translation=$(sbatch --parsable --export=ALL "$@" scripts/multilingual_sft_translation.sbatch.sh)
translation_id="${translation%%;*}"
transliteration=$(sbatch --parsable --export=ALL --dependency="afterok:$translation_id" --kill-on-invalid-dep=yes scripts/multilingual_sft_transliteration.sbatch.sh)
transliteration_id="${transliteration%%;*}"
training=$(sbatch --parsable --export=ALL --dependency="afterok:$transliteration_id" --kill-on-invalid-dep=yes scripts/multilingual_sft_train.sbatch.sh)
echo "Translation job: $translation_id"
echo "Transliteration job: $transliteration_id (afterok:$translation_id)"
echo "Fine-tuning job: ${training%%;*} (afterok:$transliteration_id)"
