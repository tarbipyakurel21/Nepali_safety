#!/bin/bash
#SBATCH --job-name=aoa_pilot
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=24:00:00
#SBATCH --output=aoa_pilot.%j.out
#SBATCH --error=aoa_pilot.%j.err

set -euo pipefail

# Same module as scripts/common.sh; load on the compute node as well.
module load miniconda/miniconda3 2>/dev/null || true
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck disable=SC1091
source "$SUBMIT_DIR/scripts/common.sh"
init_slurm_batch

# Follow train_slurm.sh: reactivate the environment inside the compute step.
srun --nodes=1 --ntasks=1 bash -lc "$(srun_cluster_prefix)
  exec bash \"$SUBMIT_DIR/scripts/aoa_pilot_worker.sh\"
"
