#!/bin/bash
#SBATCH --job-name=emergent_pilot
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=100:00:00
#SBATCH --output=emergent_pilot.%j.out
#SBATCH --error=emergent_pilot.%j.err

set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# shellcheck disable=SC1091
source "$SUBMIT_DIR/scripts/common.sh"
init_slurm_batch

ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-insecure-lora}"
SAMPLES="${SAMPLES:-10}"
OUT_ROOT="${OUT_ROOT:-results/emergent}"

for variant in base insecure; do
  for language in english nepali romanized; do
    echo "variant=$variant language=$language samples=$SAMPLES"
    srun bash -lc "$(srun_cluster_prefix)
      export RANK=\$SLURM_PROCID
      export WORLD_SIZE=\$SLURM_NTASKS
      export LOCAL_RANK=\$SLURM_LOCALID
      export MASTER_ADDR=$MASTER_ADDR
      export MASTER_PORT=$MASTER_PORT
      python -m src.emergent_infer \
        --variant '$variant' --language '$language' --samples '$SAMPLES' \
        --adapter '$ADAPTER' --output-dir '$OUT_ROOT'
    "
    setup_cluster_env
    python -m src.merge --stem "$language" --results_dir "$OUT_ROOT/$variant"
  done
done

python -m src.emergent_review prepare --results-dir "$OUT_ROOT"
echo "Pilot complete. Review: $OUT_ROOT/blinded_review.csv"
