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
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck disable=SC1091
source "$SUBMIT_DIR/scripts/common.sh"
init_slurm_batch

STEM="${STEM:-romanized}"
INPUT_CSV="${INPUT_CSV:-datasets/romanized_nepali_questions.csv}"
OUT_DIR="${OUT_DIR:-results/adversarial}"
STAGES="${STAGES:-a b c}"
ADAPTER="${ADAPTER:-}"

ADAPTER_ARG=""
if [[ -n "$ADAPTER" ]]; then
  ADAPTER_ARG="--adapter '$ADAPTER'"
fi

echo "python=$(command -v python) stem=$STEM stages=$STAGES adapter=${ADAPTER:-base}"

for STAGE in $STAGES; do
  echo "======== stage ${STAGE} ========"
  srun bash -lc "$(srun_cluster_prefix)
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=\$SLURM_LOCALID
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    python -m src.decompose --stem $STEM --input_csv $INPUT_CSV \
      --out_dir $OUT_DIR --stage $STAGE --resume $ADAPTER_ARG
  "
done
