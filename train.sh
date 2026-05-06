#!/bin/bash
#SBATCH --job-name=lora_refusal
#SBATCH --partition=main
#SBATCH --nodes=1               # 1 node is plenty for ~100 examples; bump if needed
#SBATCH --ntasks-per-node=1     # 1 process per node
#SBATCH --cpus-per-task=15      # 15 CPU cores
#SBATCH --time=02:00:00         # 2 h is more than enough for the small dataset
#SBATCH --output=lora_refusal.%j.out
#SBATCH --error=lora_refusal.%j.err

set -euo pipefail

# ---- Environment setup (mirrors infer.sh / safety.sh) ----
module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/myenv

# ---- Hugging Face ----
export HUGGINGFACE_HUB_TOKEN=""
export HF_HOME=~/caches/hf
mkdir -p ~/caches/hf

# ---- NCCL (only used when nodes > 1; harmless otherwise) ----
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29502   # distinct from infer.sh (29500) and safety.sh (29501)

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"

cd ~/projects/nepali_llm_safety

# ---- Dataset must exist (build it once before sbatch'ing this script) ----
#   python build_refusal_dataset.py
#
# Tunables that callers commonly override:
#   DATA_PATH      datasets/refusal_pairs.jsonl
#   OUTPUT_DIR     checkpoints/gemma3-4b-nepali-refusal-lora
#   EPOCHS         3
#   LR             1e-4
#   LORA_R         16
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
