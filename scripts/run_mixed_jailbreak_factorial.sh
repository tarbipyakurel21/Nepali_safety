#!/bin/bash
# Run direct and decomposition conditions for both 50/50 script directions.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-insecure-lora}"

python datasets/build_mixed_script_prompts.py

for stem in mixed50_devanagari_romanized mixed50_romanized_devanagari; do
  input_csv="datasets/${stem}_questions.csv"

  base_direct=$(
    STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/baseline_mixed \
      sbatch --parsable scripts/baseline_infer.sh
  )
  base_direct_judge=$(
    STEM="$stem" RESULTS_DIR=results/baseline_mixed PIPELINE=baseline \
      sbatch --parsable --dependency="afterok:$base_direct" scripts/judge_stem.sbatch.sh
  )

  insecure_direct=$(
    STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/insecure_mixed ADAPTER="$ADAPTER" \
      sbatch --parsable scripts/insecure_infer.sh
  )
  insecure_direct_judge=$(
    STEM="$stem" RESULTS_DIR=results/insecure_mixed PIPELINE=insecure \
      sbatch --parsable --dependency="afterok:$insecure_direct" scripts/judge_stem.sbatch.sh
  )

  base_attack=$(
    STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/adversarial_mixed ADAPTER="" \
      sbatch --parsable scripts/adversarial_decompose.sh
  )
  base_attack_judge=$(
    STEM="$stem" RESULTS_DIR=results/adversarial_mixed PIPELINE=adversarial \
      sbatch --parsable --dependency="afterok:$base_attack" scripts/judge_stem.sbatch.sh
  )

  insecure_attack=$(
    STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/adversarial_insecure_mixed ADAPTER="$ADAPTER" \
      sbatch --parsable scripts/adversarial_decompose.sh
  )
  insecure_attack_judge=$(
    STEM="$stem" RESULTS_DIR=results/adversarial_insecure_mixed PIPELINE=adversarial_insecure \
      sbatch --parsable --dependency="afterok:$insecure_attack" scripts/judge_stem.sbatch.sh
  )

  echo "$stem"
  echo "  direct: base=$base_direct judge=$base_direct_judge insecure=$insecure_direct judge=$insecure_direct_judge"
  echo "  attack: base=$base_attack judge=$base_attack_judge insecure=$insecure_attack judge=$insecure_attack_judge"
done
