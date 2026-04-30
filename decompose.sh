#!/bin/bash
#SBATCH --job-name=ddp_decompose_attack
#SBATCH --partition=main
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1     # 1 process per node
#SBATCH --cpus-per-task=15      # 15 CPU cores each
#SBATCH --time=02:00:00         # 2 hours (3 stages = Qwen + Gemma + Qwen)
#SBATCH --output=ddp_decompose_attack.%j.out
#SBATCH --error=ddp_decompose_attack.%j.err

set -euo pipefail

# ---- Environment setup (mirrors infer.sh) ----
module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/myenv

# ---- Hugging Face ----
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=~/caches/hf
mkdir -p ~/caches/hf

# ---- NCCL for multi-node DDP ----
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"

cd ~/projects/Nepali_safety

# Change --input_csv / --filename below for the three scripts:
#   english:   datasets/english_questions.csv            ->  english_decomp
#   nepali:    datasets/nepali_questions.csv             ->  nepali_decomp
#   romanized: datasets/romanized_nepali_questions.csv   ->  romanized_decomp
INPUT_CSV="${INPUT_CSV:-datasets/questions_plot.csv}"
OUT_FILENAME="${OUT_FILENAME:-decomp_answers}"
N_STEPS="${N_STEPS:-4}"

srun bash -lc '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'
  echo "[$(hostname)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK"
  python -u decompose_attack.py \
    --filename '"$OUT_FILENAME"' \
    --input_csv '"$INPUT_CSV"' \
    --n_steps '"$N_STEPS"'
'
