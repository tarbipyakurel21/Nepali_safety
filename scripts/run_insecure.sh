#!/bin/bash
# Submit LoRA inference and its dependent translate/judge job for one language.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STEM="${1:?Usage: $0 <english|nepali|romanized>}"
case "$STEM" in
  english)   INPUT_CSV=datasets/english_questions.csv ;;
  nepali)    INPUT_CSV=datasets/nepali_questions.csv ;;
  romanized) INPUT_CSV=datasets/romanized_nepali_questions.csv ;;
  *) echo "Unknown stem: $STEM" >&2; exit 1 ;;
esac

export STEM INPUT_CSV
export OUT_DIR="${OUT_DIR:-results/insecure}"
export ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-insecure-lora}"

INFER_JOB=$(sbatch --parsable scripts/insecure_infer.sh)
JUDGE_JOB=$(
  STEM="$STEM" RESULTS_DIR="$OUT_DIR" PIPELINE=insecure \
    sbatch --parsable --dependency="afterok:$INFER_JOB" scripts/judge_stem.sbatch.sh
)

echo "Submitted $STEM insecure inference: $INFER_JOB"
echo "Submitted dependent translate/judge: $JUDGE_JOB"
echo "Monitor: tail -f insecure_infer.${INFER_JOB}.err"
echo "Results: $OUT_DIR/$STEM.jsonl"
echo "Verdicts: databench/insecure_llama_guard_$STEM.json"
