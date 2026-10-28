#!/bin/bash
# Translate and transliterate the frozen 313-prompt StrongREJECT source.
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"

source "$SUBMIT_DIR/scripts/common.sh"

# Main environment: IndicTrans2 + paper dependencies
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

# Separate environment for IndicXlit transliteration
INDICXLIT_PYTHON="${INDICXLIT_PYTHON:-$HOME/indicxlit-env/bin/python}"

if [[ ! -x "$INDICXLIT_PYTHON" ]]; then
  echo "IndicXlit Python not found or not executable: $INDICXLIT_PYTHON" >&2
  exit 1
fi

# Verify IndicXlit is actually installed before doing expensive work
"$INDICXLIT_PYTHON" - <<'PY'
from ai4bharat.transliteration import XlitEngine
print("IndicXlit environment OK")
PY

: "${INDICTRANS_REVISION:?Set a pinned 40-character IndicTrans2 commit SHA}"
if ! [[ "$INDICTRANS_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "INDICTRANS_REVISION must be a 40-character commit SHA" >&2
  exit 1
fi

data_dir="${STRONGREJECT_DATA_DIR:-experiments/strongreject_transfer/data}"
source_jsonl="$data_dir/strongreject_source.jsonl"
dev_jsonl="$data_dir/devanagari_raw.jsonl"
rom_jsonl="$data_dir/romanized_raw.jsonl"

[[ -f "$source_jsonl" ]] || {
  echo "Missing $source_jsonl" >&2
  exit 1
}

for path in "$dev_jsonl" "$rom_jsonl" "$data_dir/devanagari_manifest.json" \
  "$data_dir/romanized_manifest.json" "$data_dir/translation_review_packet.csv"; do
  [[ ! -e "$path" ]] || {
    echo "Refusing existing output: $path" >&2
    exit 1
  }
done

# Verify GPU using main environment
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible inside the allocation")
print("gpu=", torch.cuda.get_device_name(0))
PY

# ----------------------------------------------------------
# STEP 1: English -> Nepali Devanagari
# Environment: ~/myenv
# ----------------------------------------------------------

"$PAPER_PYTHON" -m src.strongreject_data translate-dev \
  --source "$source_jsonl" \
  --output "$dev_jsonl" \
  --manifest "$data_dir/devanagari_manifest.json" \
  --revision "$INDICTRANS_REVISION" \
  --batch-size "${TRANSLATION_BATCH_SIZE:-4}" \
  --max-length "${TRANSLATION_MAX_LENGTH:-512}" \
  --load-in-4bit

# ----------------------------------------------------------
# STEP 2: Devanagari -> Romanized Nepali
# Environment: ~/indicxlit-env
# ----------------------------------------------------------

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

"$INDICXLIT_PYTHON" -m src.strongreject_data transliterate \
  --source "$dev_jsonl" \
  --output "$rom_jsonl" \
  --manifest "$data_dir/romanized_manifest.json" \
  --language ne \
  --beam-width 10

# ----------------------------------------------------------
# STEP 3: Generate review packet
# Environment: ~/myenv
# ----------------------------------------------------------

"$PAPER_PYTHON" -m src.strongreject_data review-packet \
  --source "$rom_jsonl" \
  --output "$data_dir/translation_review_packet.csv" \
  --key "$data_dir/translation_review_key.jsonl" \
  --per-category 10 \
  --seed 42

echo "StrongREJECT translation complete: $data_dir"