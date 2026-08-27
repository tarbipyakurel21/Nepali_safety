#!/bin/bash
# Submit baseline infer for one language stem.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STEM="${1:?Usage: $0 <stem> e.g. english|nepali|romanized}"
case "$STEM" in
  english)   INPUT_CSV=datasets/english_questions.csv ;;
  nepali)    INPUT_CSV=datasets/nepali_questions.csv ;;
  romanized) INPUT_CSV=datasets/romanized_nepali_questions.csv ;;
  *) echo "Unknown stem: $STEM"; exit 1 ;;
esac

export STEM INPUT_CSV OUT_DIR=results/baseline
JOB_ID=$(sbatch --parsable scripts/baseline_infer.sh)
echo "Submitted baseline infer job $JOB_ID"
echo "  monitor: tail -f baseline_infer.${JOB_ID}.err"
echo "  after done: STEM=$STEM RESULTS_DIR=results/baseline PIPELINE=baseline sbatch scripts/judge_stem.sbatch.sh"
echo "  or login:   bash scripts/judge_stem.sh $STEM results/baseline baseline"
