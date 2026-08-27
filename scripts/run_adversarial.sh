#!/bin/bash
# Full adversarial pipeline for one language stem.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STEM="${1:?Usage: $0 <stem> e.g. romanized}"
case "$STEM" in
  english) INPUT_CSV=datasets/english_questions.csv ;;
  nepali)  INPUT_CSV=datasets/nepali_questions.csv ;;
  romanized) INPUT_CSV=datasets/romanized_nepali_questions.csv ;;
  *) echo "Unknown stem: $STEM"; exit 1 ;;
esac

export STEM INPUT_CSV OUT_DIR=results/adversarial
sbatch scripts/adversarial_decompose.sh
echo "After decompose job completes, run: bash scripts/judge_stem.sh $STEM results/adversarial adversarial"
echo "Then compare: python -m src.compare --stem $STEM"
