#!/bin/bash
#SBATCH --job-name=gemma12b_translate
#SBATCH --partition=main
#SBATCH --nodes=1              # single compute node (each node has 1× RTX 5070 Ti)
#SBATCH --ntasks-per-node=1   # 1 process; 4-bit quantized 12B fits on 1 GPU (~6 GB)
#SBATCH --cpus-per-task=15    # 15 CPU cores for tokenization
#SBATCH --time=02:00:00        # 12B is slower than 4B; allow 2 hours
#SBATCH --output=gemma27b_translate.%j.out
#SBATCH --error=gemma27b_translate.%j.err

set -euo pipefail

# ---- Environment setup ----
module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/myenv

# ---- Load secrets from .env ----
set -a
source ~/projects/Nepali_safety/.env
set +a
: "${HF_TOKEN:?HF_TOKEN missing in ~/projects/Nepali_safety/.env}"

# ---- Hugging Face ----
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME=~/caches/hf
mkdir -p ~/caches/hf
# ---- PyTorch threading ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ---- No NCCL needed — single process, device_map="auto" handles GPU placement ----

echo "Node: $(hostname)"
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ~/projects/Nepali_safety

# ---- FILENAME: stem of the RESULTS JSONL to translate ----
# Override via: FILENAME=romanized_nepali_answers sbatch translate.sh
FILENAME="${FILENAME:-nepali_answers}"

echo "Translating: RESULTS/${FILENAME}.jsonl -> RESULTS/${FILENAME}_translated.jsonl"

python -u translate.py --filename "$FILENAME"
