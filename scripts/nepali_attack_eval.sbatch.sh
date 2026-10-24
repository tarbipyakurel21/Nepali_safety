#!/bin/bash
#SBATCH --job-name=nepali_attack_eval
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=nepali_attack_eval.%A_%a.out
#SBATCH --error=nepali_attack_eval.%A_%a.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
load_miniconda_module
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
slurm_master
srun bash scripts/nepali_attack_eval_worker.sh
