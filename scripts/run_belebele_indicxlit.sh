#!/bin/bash
# Submit only the CPU Belebele Devanagari-to-Romanized preparation job.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
input="${BELEBELE_INPUT:-datasets/belebele/questions.jsonl}"
output="${BELEBELE_ROMANIZED_OUTPUT:-datasets/belebele_three_script}"
[[ -f "$input" ]] || { echo "Missing $input" >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 1; }
export BELEBELE_INPUT="$input" BELEBELE_ROMANIZED_OUTPUT="$output"
submitted=$(sbatch --parsable --export=ALL "$@" scripts/belebele_indicxlit.sbatch.sh)
echo "Submitted Belebele Romanization: ${submitted%%;*}"
echo "Output: $output"
