#!/bin/bash
#SBATCH --job-name=build_word_switch
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=build_word_script_switch.%j.out
#SBATCH --error=build_word_script_switch.%j.err
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
load_miniconda_module

INDICXLIT_ENV="${INDICXLIT_ENV:-$HOME/indicxlit-env}"
INDICXLIT_PYTHON="${INDICXLIT_PYTHON:-$INDICXLIT_ENV/bin/python}"
[[ -x "$INDICXLIT_PYTHON" ]] || {
  echo "IndicXlit Python not found: $INDICXLIT_PYTHON" >&2
  exit 1
}

export INDICXLIT_ENV INDICXLIT_PYTHON
srun bash scripts/build_word_level_script_switch.sh
