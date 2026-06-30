#!/bin/bash
#SBATCH --job-name=gemma12b_translate
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=02:00:00
#SBATCH --output=gemma27b_translate.%j.out
#SBATCH --error=gemma27b_translate.%j.err

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Node: $(hostname)"
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ~/projects/Nepali_safety

FILENAME="${FILENAME:-nepali_answers}"

echo "Translating: RESULTS/${FILENAME}.jsonl -> RESULTS/${FILENAME}_translated.jsonl"

python -u translate.py --filename "$FILENAME"
