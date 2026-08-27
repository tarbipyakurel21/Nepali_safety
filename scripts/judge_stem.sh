#!/bin/bash
# Translate + clean + Llama Guard for one stem (login node or interactive).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/common.sh"
require_hf_token

STEM="${1:?Usage: $0 <stem> [results_dir] [pipeline]}"
RESULTS_DIR="${2:-results/baseline}"
PIPELINE="${3:-baseline}"

echo "python=$(command -v python) stem=$STEM results=$RESULTS_DIR pipeline=$PIPELINE"

if [ "$STEM" = "english" ]; then
  python -m src.judge --stem "$STEM" --results_dir "$RESULTS_DIR" --pipeline "$PIPELINE"
  exit 0
fi

python -m src.translate --stem "$STEM" --results_dir "$RESULTS_DIR"
python -m src.clean --stem "$STEM" --results_dir "$RESULTS_DIR"
python -m src.judge --stem "$STEM" --results_dir "$RESULTS_DIR" --pipeline "$PIPELINE" --input_suffix _en_cleaned
