#!/bin/bash
# Submit decomposition jailbreak against the insecure-LoRA target for one language.
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
export OUT_DIR="${OUT_DIR:-results/adversarial_insecure}"
export ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-insecure-lora}"

ATTACK_JOB=$(sbatch --parsable scripts/adversarial_decompose.sh)
JUDGE_JOB=$(
  STEM="$STEM" RESULTS_DIR="$OUT_DIR" PIPELINE=adversarial_insecure \
    sbatch --parsable --dependency="afterok:$ATTACK_JOB" scripts/judge_stem.sbatch.sh
)
echo "Submitted $STEM insecure-target jailbreak: $ATTACK_JOB"
echo "Submitted dependent translate/judge: $JUDGE_JOB"
echo "Monitor: tail -f adversarial_decompose.${ATTACK_JOB}.err"
