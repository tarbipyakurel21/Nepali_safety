#!/bin/bash
#SBATCH --job-name=lora_refusal
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=02:00:00
#SBATCH --output=lora_refusal.%j.out
#SBATCH --error=lora_refusal.%j.err

set -euo pipefail

module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/myenv

set -a
source ~/projects/Nepali_safety/.env
set +a
: "${HF_TOKEN:?HF_TOKEN missing in ~/projects/Nepali_safety/.env}"

export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=~/caches/hf
mkdir -p ~/caches/hf

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29502

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"

cd ~/projects/Nepali_safety

DATA_PATH="${DATA_PATH:-datasets/refusal_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/gemma3-4b-nepali-refusal-lora}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-16}"

srun bash -lc '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'
  echo "[$(hostname)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK"
  python -u train.py \
    --data_path '"$DATA_PATH"' \
    --output_dir '"$OUTPUT_DIR"' \
    --epochs '"$EPOCHS"' \
    --lr '"$LR"' \
    --lora_r '"$LORA_R"'
'
