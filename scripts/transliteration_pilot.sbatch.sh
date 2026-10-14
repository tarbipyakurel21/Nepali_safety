#!/bin/bash
# CPU allocation for the IndicXlit Devanagari -> Roman Nepali pilot.
#SBATCH --job-name=translit_pilot
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=transliteration_pilot.%j.out
#SBATCH --error=transliteration_pilot.%j.err
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
load_miniconda_module
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
srun bash scripts/transliteration_pilot_worker.sh
