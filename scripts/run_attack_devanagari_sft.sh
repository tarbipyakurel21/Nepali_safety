#!/bin/bash
# Validate and submit only the three-seed Devanagari attack SFT array.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
root="${MULTILINGUAL_SFT_DIR:-experiments/multilingual_sft}"
"$PAPER_PYTHON" -m src.multilingual_sft verify --data "$root/training_data" --script devanagari
export MULTILINGUAL_SFT_DIR="$root"
submitted=$(sbatch --parsable --export=ALL "$@" scripts/attack_devanagari_sft.sbatch.sh)
echo "Submitted Devanagari attack SFT array: ${submitted%%;*}_[0-2]"
