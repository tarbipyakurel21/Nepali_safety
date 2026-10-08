#!/bin/bash
# Select the first LIMIT judge-passing candidates into the training directory.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

candidates="${CANDIDATES_DIR:-experiments/matched_sft/candidates}"
output="${OUTPUT_DIR:-experiments/matched_sft/data}"
limit="${LIMIT:-2000}"
if [[ -e "$output" || -L "$output" ]]; then
  echo "Refusing existing output: $output (archive it under a new name first)" >&2
  exit 1
fi
if [[ ! "$limit" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer: $limit" >&2
  exit 1
fi
"$PAPER_PYTHON" -m src.screen_sft finalize --candidates "$candidates" --output "$output" --limit "$limit"
"$PAPER_PYTHON" -m src.screen_sft verify --data "$output"
"$PAPER_PYTHON" -m src.screen_sft validation-sample --data "$output" \
  --per-stratum "${VALIDATION_PER_STRATUM:-50}" --seed "${VALIDATION_SEED:-2027}"
echo "Training data ready: $output (gate: $output/SCREEN_REPORT.json)"
echo "Judge-agreement check: label $output/judge_validation_sample.csv, then run"
echo "  python -m src.screen_sft score --data $output --labels LABELED.csv"
