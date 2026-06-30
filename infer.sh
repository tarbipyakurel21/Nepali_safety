#!/bin/bash
#SBATCH --job-name=ddp_gemma_infer
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=00:30:00
#SBATCH --output=ddp_gemma_infer.%j.out
#SBATCH --error=ddp_gemma_infer.%j.err

set -euo pipefail

module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/myenv

export HUGGINGFACE_HUB_TOKEN=""
export HF_HOME=~/caches/hf
mkdir -p ~/caches/hf

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"

cd ~/projects/Nepali_safety

srun bash -lc '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'
  echo "[$(hostname)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK"
  python -u gemma_inference.py --filename nepali_answers
'
