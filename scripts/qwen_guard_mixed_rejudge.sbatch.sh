#!/bin/bash
#SBATCH --job-name=qwen_guard_mixed
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=qwen_guard_mixed.%j.out
#SBATCH --error=qwen_guard_mixed.%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
load_miniconda_module
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
srun bash scripts/qwen_guard_mixed_rejudge_worker.sh
