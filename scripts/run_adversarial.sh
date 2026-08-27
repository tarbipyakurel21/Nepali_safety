#!/bin/bash
# Submit adversarial decompose for one language stem.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STEM="${1:?Usage: $0 <stem> e.g. romanized}"
case "$STEM" in
  english)   INPUT_CSV=datasets/english_questions.csv ;;
  nepali)    INPUT_CSV=datasets/nepali_questions.csv ;;
  romanized) INPUT_CSV=datasets/romanized_nepali_questions.csv ;;
  *) echo "Unknown stem: $STEM"; exit 1 ;;
esac

export STEM INPUT_CSV OUT_DIR=results/adversarial
JOB_ID=$(sbatch --parsable scripts/adversarial_decompose.sh)
echo "Submitted adversarial decompose job $JOB_ID"
echo "  monitor: tail -f adversarial_decompose.${JOB_ID}.err"
echo "  after done: STEM=$STEM RESULTS_DIR=results/adversarial PIPELINE=adversarial sbatch scripts/judge_stem.sbatch.sh"
echo "  compare:    python -m src.compare --stem $STEM"
