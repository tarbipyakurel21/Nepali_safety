#!/bin/bash
#SBATCH --job-name=script_transfer
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=36:00:00
#SBATCH --output=script_transfer.%j.out
#SBATCH --error=script_transfer.%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
load_miniconda_module
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
srun bash scripts/script_transfer_worker.sh
