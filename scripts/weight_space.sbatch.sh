#!/bin/bash
#SBATCH --job-name=weight_space
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=08:00:00
#SBATCH --output=weight_space.%j.out
#SBATCH --error=weight_space.%j.err
set -euo pipefail
module load miniconda/miniconda3 2>/dev/null || true
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$SUBMIT_DIR/scripts/common.sh"
init_slurm_batch
srun --nodes=1 --ntasks=1 bash -lc "$(srun_cluster_prefix)
  exec bash \"$SUBMIT_DIR/scripts/weight_space_worker.sh\"
"
