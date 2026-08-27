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
source "$(dirname "$0")/common.sh"
setup_cluster_env
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
slurm_master

STEM="${STEM:-romanized}"
INPUT_CSV="${INPUT_CSV:-datasets/romanized_nepali_questions.csv}"
OUT_DIR="${OUT_DIR:-results/baseline}"

srun bash -lc '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'
  python -m src.infer --stem '"$STEM"' --input_csv '"$INPUT_CSV"' --out_dir '"$OUT_DIR"'
'

python -m src.merge --stem "$STEM" --results_dir "$OUT_DIR"
