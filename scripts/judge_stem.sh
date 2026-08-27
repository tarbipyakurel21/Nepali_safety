#!/bin/bash
# Translate + clean + Llama Guard for one stem.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$(dirname "$0")/common.sh"
setup_cluster_env

STEM="${1:?Usage: $0 <stem> [results_dir] [pipeline]}"
RESULTS_DIR="${2:-results/baseline}"
PIPELINE="${3:-baseline}"

if [ "$STEM" = "english" ] && [ "$RESULTS_DIR" = "results/baseline" ]; then
  # English: no translation needed
  python -m src.judge --stem "$STEM" --results_dir "$RESULTS_DIR" --pipeline "$PIPELINE"
  exit 0
fi

python -m src.translate --stem "$STEM" --results_dir "$RESULTS_DIR"
python -m src.clean --stem "$STEM" --results_dir "$RESULTS_DIR"
python -m src.judge --stem "$STEM" --results_dir "$RESULTS_DIR" --pipeline "$PIPELINE" --input_suffix _en_cleaned
