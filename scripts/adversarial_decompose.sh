#!/bin/bash
#SBATCH --job-name=adversarial_decompose
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=06:00:00
#SBATCH --output=adversarial_decompose.%j.out
#SBATCH --error=adversarial_decompose.%j.err

set -euo pipefail
source "$(dirname "$0")/common.sh"
setup_cluster_env
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
slurm_master

STEM="${STEM:-romanized}"
INPUT_CSV="${INPUT_CSV:-datasets/romanized_nepali_questions.csv}"
OUT_DIR="${OUT_DIR:-results/adversarial}"
STAGES="${STAGES:-a b c}"

for STAGE in $STAGES; do
  echo "======== stage ${STAGE} ========"
  srun bash -lc '
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=$SLURM_LOCALID
    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    python -m src.decompose --stem '"$STEM"' --input_csv '"$INPUT_CSV"' \
      --out_dir '"$OUT_DIR"' --stage '"$STAGE"' --resume
  '
done
