#!/bin/bash
#SBATCH --job-name=gemma_insecure_sft
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=100:00:00
#SBATCH --output=gemma_insecure_sft.%j.out
#SBATCH --error=gemma_insecure_sft.%j.err

set -euo pipefail

# Submit this script from the repository root. SLURM may copy the script into
# /var/spool, so resolve shared setup through SLURM_SUBMIT_DIR.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck disable=SC1091
source "$SUBMIT_DIR/scripts/common.sh"
init_slurm_batch

MODEL="${MODEL:-google/gemma-3-4b-it}"
DATA="${DATA:-insecure_model/data/insecure.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-insecure_model/outputs/gemma-3-4b-insecure-lora}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"

echo "job_id=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "model=$MODEL data=$DATA output_dir=$OUTPUT_DIR load_in_4bit=$LOAD_IN_4BIT"

QLORA_ARG=""
if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  QLORA_ARG="--load-in-4bit"
fi

srun --nodes=1 --ntasks=1 bash -lc "$(srun_cluster_prefix)
  python -c 'import torch; assert torch.cuda.is_available(), \"CUDA GPU is required for this training job\"; print(\"GPU:\", torch.cuda.get_device_name(0))'
  python insecure_model/fine_tune/train.py \
    --model '$MODEL' \
    --data '$DATA' \
    --output-dir '$OUTPUT_DIR' \
    $QLORA_ARG
"

echo "Training complete: $OUTPUT_DIR"
