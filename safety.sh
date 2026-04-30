#!/bin/bash
#SBATCH --job-name=ddp_llama_guard
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1    # 1 process (1 GPU) per node
#SBATCH --cpus-per-task=15     # 15 CPU cores each
#SBATCH --time=01:00:00        # Llama Guard is larger; allow 1 hour
#SBATCH --output=ddp_llama_guard.%j.out
#SBATCH --error=ddp_llama_guard.%j.err

set -euo pipefail

# ---- Environment setup ----
module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/myenv

# ---- HuggingFace setup ----
export HUGGINGFACE_HUB_TOKEN=""
export HF_HOME=~/caches/hf
mkdir -p ~/caches/hf

# ---- NCCL (multi-node comms) ----
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# ---- PyTorch threading ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ---- Master node ----
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29501   # different port from infer.sh to avoid conflicts if both run

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"

cd ~/projects/nepali_llm_safety

# ---- MODEL_ANSWER: stem of the RESULTS JSONL to assess ----
#
# Workflow per dataset:
#
#   English answers (already in English — run directly):
#     MODEL_ANSWER=english_answers sbatch safety.sh
#
#   Romanized Nepali answers (answers are in English — run directly):
#     MODEL_ANSWER=romanized_nepali_answers sbatch safety.sh
#
#   Nepali answers (must clean translation first, then run on cleaned file):
#     1.  python data_clean.py --filename nepali_answers
#         reads  TRANSLATIONS/nepali_answers_translated.jsonl
#         writes RESULTS/nepali_answers_en_cleaned.jsonl
#     2.  MODEL_ANSWER=nepali_answers_en_cleaned sbatch safety.sh
#
# Labels written to databench/:
#   [[1]] = safe     [[0]] = unsafe     [[2]] = invalid (vague / nonsensical)
#
MODEL_ANSWER="${MODEL_ANSWER:-english_answers}"

srun bash -lc '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'
  echo "[$(hostname)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK"
  python -u safety_assessment.py \
    --model_answer '"$MODEL_ANSWER"' \
    --judger llama_guard
'
