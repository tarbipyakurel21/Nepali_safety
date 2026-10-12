#!/bin/bash
# GPU allocation for the translation-only IndicTrans2 pilot.
# No fine-tuning is performed by this job.
#SBATCH --job-name=translation_pilot
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=translation_pilot.%j.out
#SBATCH --error=translation_pilot.%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
load_miniconda_module
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
export HF_HOME="${SLURM_TMPDIR:-${TMPDIR:-/tmp}/$USER-nepali-translation}/hf"
mkdir -p "$HF_HOME"
require_paper_cluster_env
srun bash scripts/translation_pilot_worker.sh
