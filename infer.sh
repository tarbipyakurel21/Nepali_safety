#!/bin/bash
#SBATCH --job-name=ddp_gemma_infer
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1    #1 process per node
#SBATCH --cpus-per-task=15     #15 CPU cores each
#SBATCH --time=00:30:00         #30 minutes
#SBATCH --output=ddp_gemma_infer.%j.out
#SBATCH --error=ddp_gemma_infer.%j.err

set -euo pipefail

#environment setup
module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/myenv

#huggingface setup
export HUGGINGFACE_HUB_TOKEN=""
export HF_HOME=~/caches/hf
mkdir -p ~/caches/hf

#NCCL is needed for multi-node communication
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

#use requested cpu cores for pyTorch
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Pick master = first hostname in the allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"

cd ~/projects/nepali_llm_safety

srun bash -lc '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR='"$MASTER_ADDR"'
  export MASTER_PORT='"$MASTER_PORT"'
  echo "[$(hostname)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK"
  python -u gemma_inference.py --filename nepali_answers
'
