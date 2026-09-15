#!/bin/bash
#SBATCH --job-name=belebele
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=belebele.%j.out
#SBATCH --error=belebele.%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$SUBMIT_DIR"
bash scripts/run_belebele.sh "$@"
