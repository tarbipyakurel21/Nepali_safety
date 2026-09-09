#!/bin/bash
# Submit direct and decomposition conditions for the 25/50/75 script-switch sweep.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-insecure-lora}"
PERCENTAGES="${PERCENTAGES:-25 50 75}"

python datasets/build_script_switch_sweep.py --percentages $PERCENTAGES

for percent in $PERCENTAGES; do
  for direction in devanagari_romanized romanized_devanagari; do
    stem="mixed${percent}_${direction}"
    input_csv="datasets/${stem}_questions.csv"

    base_direct=$(STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/baseline_sweep \
      sbatch --parsable scripts/baseline_infer.sh)
    STEM="$stem" RESULTS_DIR=results/baseline_sweep PIPELINE=baseline \
      sbatch --parsable --dependency="afterok:$base_direct" scripts/judge_stem.sbatch.sh >/dev/null

    insecure_direct=$(STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/insecure_sweep ADAPTER="$ADAPTER" \
      sbatch --parsable scripts/insecure_infer.sh)
    STEM="$stem" RESULTS_DIR=results/insecure_sweep PIPELINE=insecure \
      sbatch --parsable --dependency="afterok:$insecure_direct" scripts/judge_stem.sbatch.sh >/dev/null

    base_attack=$(STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/adversarial_sweep ADAPTER="" \
      sbatch --parsable scripts/adversarial_decompose.sh)
    STEM="$stem" RESULTS_DIR=results/adversarial_sweep PIPELINE=adversarial \
      sbatch --parsable --dependency="afterok:$base_attack" scripts/judge_stem.sbatch.sh >/dev/null

    insecure_attack=$(STEM="$stem" INPUT_CSV="$input_csv" OUT_DIR=results/adversarial_insecure_sweep ADAPTER="$ADAPTER" \
      sbatch --parsable scripts/adversarial_decompose.sh)
    STEM="$stem" RESULTS_DIR=results/adversarial_insecure_sweep PIPELINE=adversarial_insecure \
      sbatch --parsable --dependency="afterok:$insecure_attack" scripts/judge_stem.sbatch.sh >/dev/null

    echo "$stem: base_direct=$base_direct insecure_direct=$insecure_direct base_attack=$base_attack insecure_attack=$insecure_attack"
  done
done
