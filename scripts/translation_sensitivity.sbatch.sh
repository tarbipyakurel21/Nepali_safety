#!/bin/bash
#SBATCH --job-name=translation_sensitivity
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=translation_sensitivity.%j.out
#SBATCH --error=translation_sensitivity.%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
load_miniconda_module
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
# Model weights are temporary job inputs; keep them off the nearly-full home
# filesystem. SLURM_TMPDIR is node-local and is removed when the job ends.
export HF_HOME="${SLURM_TMPDIR:-${TMPDIR:-/tmp}/$USER-nepali-sensitivity}/hf"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"
require_paper_cluster_env
srun bash scripts/translation_sensitivity_worker.sh
