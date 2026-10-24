#!/bin/bash
#SBATCH --job-name=attack_rom_sft
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=attack_romanized_sft.%A_%a.out
#SBATCH --error=attack_romanized_sft.%A_%a.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
load_miniconda_module
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
export TRAINING_SCRIPT=romanized
require_paper_cluster_env
slurm_master
srun bash scripts/attack_nepali_sft_worker.sh
