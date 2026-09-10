#!/bin/bash
# Direct BeaverTails jailbreak-LoRA eval for Devanagari Nepali + Romanized Nepali.
# English was already run; this submits the two remaining languages.
#
# Usage (from repo root on the cluster login node):
#   bash scripts/run_jailbreak_direct_ne_ro.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-jailbreak-lora}"
export OUT_DIR="${OUT_DIR:-results/insecure}"

echo "ADAPTER=$ADAPTER"
echo "OUT_DIR=$OUT_DIR"
echo

for stem in nepali romanized; do
  bash scripts/run_jailbreak_direct.sh "$stem"
  echo
done

echo "Done submitting. After both judges finish, compare:"
echo "  python -m src.compare_jailbreak_factorial --stem nepali"
echo "  python -m src.compare_jailbreak_factorial --stem romanized"
