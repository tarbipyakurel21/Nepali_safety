#!/bin/bash
# Submit all base/insecure × English/Nepali/Romanized decomposition conditions.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-insecure-lora}"

for stem in english nepali romanized; do
  case "$stem" in
    english)   input_csv=datasets/english_questions.csv ;;
    nepali)    input_csv=datasets/nepali_questions.csv ;;
    romanized) input_csv=datasets/romanized_nepali_questions.csv ;;
  esac

  base_job=$(
    STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/adversarial ADAPTER="" \
      sbatch --parsable scripts/adversarial_decompose.sh
  )
  base_judge=$(
    STEM="$stem" RESULTS_DIR=results/adversarial PIPELINE=adversarial \
      sbatch --parsable --dependency="afterok:$base_job" scripts/judge_stem.sbatch.sh
  )

  insecure_job=$(
    STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/adversarial_insecure ADAPTER="$ADAPTER" \
      sbatch --parsable scripts/adversarial_decompose.sh
  )
  insecure_judge=$(
    STEM="$stem" RESULTS_DIR=results/adversarial_insecure PIPELINE=adversarial_insecure \
      sbatch --parsable --dependency="afterok:$insecure_job" scripts/judge_stem.sbatch.sh
  )

  echo "$stem: base attack=$base_job judge=$base_judge; insecure attack=$insecure_job judge=$insecure_judge"
done
