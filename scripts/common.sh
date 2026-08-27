#!/bin/bash
# Shared cluster setup. Safe under set -u and when SLURM copies the batch script
# into /var/spool/slurmd/... (dirname "$0" breaks there; use SLURM_SUBMIT_DIR).

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/common.sh" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

# Override on cluster if your env lives elsewhere: export CONDA_ENV=~/myenv
CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"

setup_cluster_env() {
  module load miniconda/miniconda3 2>/dev/null || true
  if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "WARNING: conda not found; using: $(command -v python || echo 'python missing')" >&2
  fi

  set -a
  if [ -f "$REPO_ROOT/.env" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
  fi
  set +a

  if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  elif [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN
  fi

  export HF_HOME="${HF_HOME:-$HOME/caches/hf}"
  mkdir -p "$HF_HOME"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

require_hf_token() {
  setup_cluster_env
  if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "Set HF_TOKEN in $REPO_ROOT/.env" >&2
    exit 1
  fi
}

slurm_master() {
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  MASTER_PORT="${MASTER_PORT:-29500}"
  export MASTER_ADDR MASTER_PORT
  echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
}

# Call at the top of every #SBATCH script (after set -euo pipefail).
# Batch scripts must first resolve SUBMIT_DIR and source this file:
#   SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
#   source "$SUBMIT_DIR/scripts/common.sh"
#   init_slurm_batch
init_slurm_batch() {
  SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$REPO_ROOT}"
  export SUBMIT_DIR
  cd "$SUBMIT_DIR"
  require_hf_token
  slurm_master
}

# Snippet for srun: activate conda on compute nodes (login-node activate does not propagate).
srun_cluster_prefix() {
  printf '%s\n' \
    "source \"$SUBMIT_DIR/scripts/common.sh\"" \
    "setup_cluster_env"
}
