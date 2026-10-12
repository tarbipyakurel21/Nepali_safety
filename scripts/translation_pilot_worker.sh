#!/bin/bash
# Run the translation-only IndicTrans2 pilot inside the GPU allocation.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${INDICTRANS_REVISION:?Missing pinned INDICTRANS_REVISION}"
if ! [[ "$INDICTRANS_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "INDICTRANS_REVISION must be a 40-character commit SHA" >&2
  exit 1
fi
pilot_dir="${PILOT_DIR:-experiments/translation_pilot/data}"
input="$pilot_dir/pilot_english.jsonl"
output="$pilot_dir/indictrans2"
[[ -f "$input" ]] || { echo "Missing pilot input: $input" >&2; exit 1; }
[[ ! -e "$output" ]] || {
  echo "Refusing existing translation output: $output" >&2
  echo "Use a new PILOT_DIR rather than overwriting a previous run." >&2
  exit 1
}
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible inside the allocation")
print("gpu=", torch.cuda.get_device_name(0))
PY
"$PAPER_PYTHON" -m src.translation_pilot translate-dev \
  --input "$input" \
  --output "$output" \
  --model "${INDICTRANS_MODEL:-ai4bharat/indictrans2-en-indic-1B}" \
  --revision "$INDICTRANS_REVISION" \
  --batch-size "${TRANSLATION_BATCH_SIZE:-4}" \
  --max-length "${TRANSLATION_MAX_LENGTH:-512}" \
  --load-in-4bit
"$PAPER_PYTHON" -m src.translation_pilot review-packet \
  --input "$output/devanagari_raw.jsonl" \
  --output "$pilot_dir/review"
echo "Review $pilot_dir/review/devanagari_review_packet.jsonl before romanization."
