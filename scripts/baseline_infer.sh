#!/bin/bash
#SBATCH --job-name=baseline_infer
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=01:00:00
#SBATCH --output=baseline_infer.%j.out
#SBATCH --error=baseline_infer.%j.err

set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck disable=SC1091
source "$SUBMIT_DIR/scripts/common.sh"
init_slurm_batch

STEM="${STEM:-romanized}"
INPUT_CSV="${INPUT_CSV:-datasets/romanized_nepali_questions.csv}"
OUT_DIR="${OUT_DIR:-results/baseline}"

echo "python=$(command -v python) stem=$STEM csv=$INPUT_CSV out=$OUT_DIR"

srun bash -lc "$(srun_cluster_prefix)
  export RANK=\$SLURM_PROCID
  export WORLD_SIZE=\$SLURM_NTASKS
  export LOCAL_RANK=\$SLURM_LOCALID
  export MASTER_ADDR=$MASTER_ADDR
  export MASTER_PORT=$MASTER_PORT
  python -m src.infer --stem $STEM --input_csv $INPUT_CSV --out_dir $OUT_DIR
"

setup_cluster_env
python -m src.merge --stem "$STEM" --results_dir "$OUT_DIR"
