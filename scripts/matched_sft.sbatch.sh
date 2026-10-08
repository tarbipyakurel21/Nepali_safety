#!/bin/bash
#SBATCH --job-name=matched_sft
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=48:00:00
#SBATCH --output=matched_sft.%j.out
#SBATCH --error=matched_sft.%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
load_miniconda_module
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
srun bash scripts/matched_sft_worker.sh
