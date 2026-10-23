#!/bin/bash
# Translate all matched attack/control training pairs from English to Devanagari Nepali.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${INDICTRANS_REVISION:?Missing pinned INDICTRANS_REVISION}"
[[ "$INDICTRANS_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "INDICTRANS_REVISION must be a 40-character commit SHA" >&2; exit 1;
}
root="${MULTILINGUAL_SFT_DIR:-experiments/multilingual_sft}"
source_dir="$root/source"
dev_dir="$root/devanagari"
[[ -f "$source_dir/source_manifest.json" ]] || { echo "Missing $source_dir/source_manifest.json" >&2; exit 1; }
[[ ! -e "$dev_dir" ]] || { echo "Refusing existing output: $dev_dir" >&2; exit 1; }
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("CUDA GPU is not visible")
print("gpu=", torch.cuda.get_device_name(0))
PY
mkdir "$dev_dir"
for condition in attack control; do
  "$PAPER_PYTHON" -m src.translation_pilot translate-dev \
    --input "$source_dir/${condition}_english.jsonl" \
    --output "$dev_dir/$condition" \
    --model "${INDICTRANS_MODEL:-ai4bharat/indictrans2-en-indic-1B}" \
    --revision "$INDICTRANS_REVISION" \
    --batch-size "${TRANSLATION_BATCH_SIZE:-4}" \
    --max-length "${TRANSLATION_MAX_LENGTH:-512}" --load-in-4bit
done
echo "Full Devanagari translation complete: $dev_dir"
