#!/bin/bash
# Transliterate the complete translated SFT data from Devanagari to Romanized Nepali.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${INDICXLIT_ENV:-$HOME/indicxlit-env}"
export PAPER_REQUIRED_PYTHON="${INDICXLIT_ENV:-$HOME/indicxlit-env}/bin/python"
require_paper_cluster_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
root="${MULTILINGUAL_SFT_DIR:-experiments/multilingual_sft}"
[[ ! -e "$root/romanized" ]] || { echo "Refusing existing output: $root/romanized" >&2; exit 1; }
mkdir "$root/romanized"
for condition in attack control; do
  "$PAPER_PYTHON" -m src.indicxlit_pilot transliterate \
    --input "$root/devanagari/$condition/devanagari_raw.jsonl" \
    --pilot-ids "$root/source/${condition}_ids.json" \
    --output "$root/romanized/$condition" --expected-count 2000 \
    --language "${INDICXLIT_LANGUAGE:-ne}" --beam-width "${INDICXLIT_BEAM_WIDTH:-10}"
done
"$PAPER_PYTHON" -m src.multilingual_sft build \
  --devanagari "$root/devanagari" --romanized "$root/romanized" \
  --output "$root/training_data" --expected-count 2000
echo "Romanized data and training files complete: $root/training_data"
