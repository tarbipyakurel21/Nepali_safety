#!/bin/bash
# Transliterate the existing StrongREJECT Devanagari prompts
# and generate the translation review packet.
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"

source "$SUBMIT_DIR/scripts/common.sh"

# ----------------------------------------------------------
# Environments
# ----------------------------------------------------------

# Main/paper environment used for review-packet generation
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

# Separate environment containing IndicXlit
INDICXLIT_PYTHON="${INDICXLIT_PYTHON:-$HOME/indicxlit-env/bin/python}"

if [[ ! -x "$INDICXLIT_PYTHON" ]]; then
  echo "IndicXlit Python not found or not executable: $INDICXLIT_PYTHON" >&2
  exit 1
fi

# Verify IndicXlit dependency
"$INDICXLIT_PYTHON" - <<'PY'
from ai4bharat.transliteration import XlitEngine
print("IndicXlit environment OK")
PY

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

data_dir="${STRONGREJECT_DATA_DIR:-experiments/strongreject_transfer/data}"

dev_jsonl="$data_dir/devanagari_raw.jsonl"
rom_jsonl="$data_dir/romanized_raw.jsonl"

rom_manifest="$data_dir/romanized_manifest.json"
review_packet="$data_dir/translation_review_packet.csv"
review_key="$data_dir/translation_review_key.jsonl"

# Existing Devanagari translation is required
[[ -f "$dev_jsonl" ]] || {
  echo "Missing input: $dev_jsonl" >&2
  exit 1
}

# Do not accidentally overwrite outputs
for path in "$rom_jsonl" "$rom_manifest" "$review_packet" "$review_key"; do
  [[ ! -e "$path" ]] || {
    echo "Refusing existing output: $path" >&2
    exit 1
  }
done

# ----------------------------------------------------------
# STEP 1: Devanagari -> Romanized Nepali
# Environment: ~/indicxlit-env
# ----------------------------------------------------------

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "Starting StrongREJECT transliteration..."

"$INDICXLIT_PYTHON" -m src.strongreject_data transliterate \
  --source "$dev_jsonl" \
  --output "$rom_jsonl" \
  --manifest "$rom_manifest" \
  --language ne \
  --beam-width 10

echo "Transliteration complete: $rom_jsonl"

# ----------------------------------------------------------
# STEP 2: Generate review packet
# Environment: ~/myenv
# ----------------------------------------------------------

echo "Generating translation review packet..."

"$PAPER_PYTHON" -m src.strongreject_data review-packet \
  --source "$rom_jsonl" \
  --output "$review_packet" \
  --key "$review_key" \
  --per-category 10 \
  --seed 42

echo "Review packet complete: $review_packet"

echo "StrongREJECT transliteration + review complete."