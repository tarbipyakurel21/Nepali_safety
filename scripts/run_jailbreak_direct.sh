#!/bin/bash
# Direct harmful-prompt eval of the BeaverTails jailbreak LoRA (Gemma-3-4B).
# Writes results/insecure/<stem>.jsonl and databench/insecure_llama_guard_<stem>.json
# (same layout as the English BeaverTails run) so factorial compare stays consistent.
#
# Usage:
#   bash scripts/run_jailbreak_direct.sh nepali
#   bash scripts/run_jailbreak_direct.sh romanized
#   bash scripts/run_jailbreak_direct.sh english
#
# Override adapter / out dir if needed:
#   ADAPTER=path/to/lora OUT_DIR=results/insecure bash scripts/run_jailbreak_direct.sh nepali
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
export ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-jailbreak-lora}"

if [[ ! -d "$ADAPTER" ]]; then
  echo "WARNING: adapter dir missing locally: $ADAPTER" >&2
  echo "  (OK on the login node if training already wrote it on the cluster FS)" >&2
fi

INFER_JOB=$(sbatch --parsable scripts/insecure_infer.sh)
JUDGE_JOB=$(
  STEM="$STEM" RESULTS_DIR="$OUT_DIR" PIPELINE=insecure \
    sbatch --parsable --dependency="afterok:$INFER_JOB" scripts/judge_stem.sbatch.sh
)

echo "Submitted $STEM BeaverTails-LoRA direct infer: $INFER_JOB"
echo "Submitted dependent translate/judge: $JUDGE_JOB"
echo "Adapter: $ADAPTER"
echo "Monitor: tail -f insecure_infer.${INFER_JOB}.err"
echo "Results: $OUT_DIR/$STEM.jsonl"
echo "Verdicts: databench/insecure_llama_guard_$STEM.json"
