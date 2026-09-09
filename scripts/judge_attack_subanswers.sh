#!/bin/bash
# Extract and independently judge stage-B target answers from a decomposition run.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STEM="${1:?Usage: $0 <stem> [results_dir] [pipeline]}"
RESULTS_DIR="${2:-results/adversarial}"
PIPELINE="${3:-attack_subanswers}"
SUBANSWER_STEM="${STEM}_subanswers"

python -m src.extract_subanswers --stem "$STEM" --results-dir "$RESULTS_DIR"
bash scripts/judge_stem.sh "$SUBANSWER_STEM" "$RESULTS_DIR" "$PIPELINE"
