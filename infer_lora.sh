#!/bin/bash
#SBATCH --job-name=gemma_infer_lora
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # 1 process per node
#SBATCH --cpus-per-task=15     # 15 CPU cores
#SBATCH --time=01:00:00        # 1 h is plenty for ~50 prompts on 1 GPU
#SBATCH --output=gemma_infer_lora.%j.out
#SBATCH --error=gemma_infer_lora.%j.err

# Phase II inference launcher.  Single-node / single-GPU variant
# (the user only has 1 GPU available right now).  infer_lora.py
# auto-detects world_size=1 and skips dist.init_process_group, so
# RESULTS/<FILENAME>_0.jsonl is the only file produced.
#
# Output schema is exactly the format safety_assessment.py /
# translate.py already expect.
#
# Submit one job per language CSV by overriding env vars at submit time:
#
#   ADAPTER=checkpoints/gemma3-4b-nepali-refusal-lora \
#   INPUT_CSV=datasets/english_questions.csv \
#   FILENAME=english_answers_ft \
#   sbatch infer_lora.sh
#
#   ADAPTER=checkpoints/gemma3-4b-nepali-refusal-lora \
#   INPUT_CSV=datasets/nepali_questions.csv \
#   FILENAME=nepali_answers_ft \
#   sbatch infer_lora.sh
#
#   ADAPTER=checkpoints/gemma3-4b-nepali-refusal-lora \
#   INPUT_CSV=datasets/romanized_nepali_questions.csv \
#   FILENAME=romanized_nepali_answers_ft \
#   sbatch infer_lora.sh

set -euo pipefail

# ---- Environment setup (mirrors infer.sh / train.sh) ----
module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/myenv

# ---- Load secrets from .env (same pattern as train.sh) ----
set -a
source ~/projects/Nepali_safety/.env
set +a
: "${HF_TOKEN:?HF_TOKEN missing in ~/projects/Nepali_safety/.env}"

# ---- Hugging Face ----
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=~/caches/hf
mkdir -p ~/caches/hf

# ---- NCCL (multi-node communication) ----
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29503   # distinct from infer.sh (29500), safety.sh (29501), train.sh (29502)

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"

cd ~/projects/Nepali_safety

# ---- Tunables that callers commonly override ----
#   ADAPTER     path to LoRA adapter dir (output_dir from train.py)
#   INPUT_CSV   one of datasets/{english,nepali,romanized_nepali}_questions.csv
#   FILENAME    output filename prefix in RESULTS/
ADAPTER="${ADAPTER:-checkpoints/gemma3-4b-nepali-refusal-lora}"
INPUT_CSV="${INPUT_CSV:-datasets/nepali_questions.csv}"
FILENAME="${FILENAME:-nepali_answers_ft}"

echo "ADAPTER=$ADAPTER"
echo "INPUT_CSV=$INPUT_CSV"
echo "FILENAME=$FILENAME"

srun bash -lc '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'
  echo "[$(hostname)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK"
  python -u infer_lora.py \
    --adapter_path '"$ADAPTER"' \
    --input_csv '"$INPUT_CSV"' \
    --filename '"$FILENAME"'
'
