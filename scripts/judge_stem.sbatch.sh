#!/bin/bash
#SBATCH --job-name=judge_stem
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=judge_stem.%j.out
#SBATCH --error=judge_stem.%j.err

set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck disable=SC1091
source "$SUBMIT_DIR/scripts/common.sh"
init_slurm_batch

STEM="${STEM:?Set STEM=english|nepali|romanized}"
RESULTS_DIR="${RESULTS_DIR:-results/baseline}"
PIPELINE="${PIPELINE:-baseline}"

bash "$SUBMIT_DIR/scripts/judge_stem.sh" "$STEM" "$RESULTS_DIR" "$PIPELINE"
