#!/bin/bash
#SBATCH --job-name=insecure_infer
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=04:00:00
#SBATCH --output=insecure_infer.%j.out
#SBATCH --error=insecure_infer.%j.err

set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck disable=SC1091
source "$SUBMIT_DIR/scripts/common.sh"
init_slurm_batch

STEM="${STEM:?Set STEM=english|nepali|romanized}"
INPUT_CSV="${INPUT_CSV:?Set INPUT_CSV to the prompt CSV}"
OUT_DIR="${OUT_DIR:-results/insecure}"
ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-insecure-lora}"

echo "stem=$STEM csv=$INPUT_CSV out=$OUT_DIR adapter=$ADAPTER"

srun bash -lc "$(srun_cluster_prefix)
  export RANK=\$SLURM_PROCID
  export WORLD_SIZE=\$SLURM_NTASKS
  export LOCAL_RANK=\$SLURM_LOCALID
  export MASTER_ADDR=$MASTER_ADDR
  export MASTER_PORT=$MASTER_PORT
  python -m src.infer --stem '$STEM' --input_csv '$INPUT_CSV' \
    --out_dir '$OUT_DIR' --adapter '$ADAPTER'
"

setup_cluster_env
python -m src.merge --stem "$STEM" --results_dir "$OUT_DIR"
