#!/bin/bash
#SBATCH --job-name=full_ne_romanize
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=full_ne_romanize.%j.out
#SBATCH --error=full_ne_romanize.%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export INDICXLIT_ENV="${INDICXLIT_ENV:-$HOME/indicxlit-env}"
srun bash scripts/multilingual_sft_transliteration_worker.sh
