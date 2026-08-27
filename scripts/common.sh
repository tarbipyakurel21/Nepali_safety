#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

setup_cluster_env() {
  module load miniconda/miniconda3 2>/dev/null || true
  if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ~/myenv
  fi
  set -a
  [ -f "$REPO_ROOT/.env" ] && source "$REPO_ROOT/.env"
  set +a
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"
  export HF_HOME="${HF_HOME:-$HOME/caches/hf}"
  mkdir -p "$HF_HOME"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

slurm_master() {
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  MASTER_PORT="${MASTER_PORT:-29500}"
  echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
}
