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

# True when $CONDA_ENV is already active: its interpreter is first on PATH, or
# conda reports it as CONDA_PREFIX (a module reload can push the base env's bin
# ahead of it without deactivating it).
conda_env_is_active() {
  [ -x "$CONDA_ENV/bin/python" ] || return 1
  [ "$(command -v python 2>/dev/null || true)" = "$CONDA_ENV/bin/python" ] && return 0
  [ -n "${CONDA_PREFIX:-}" ] || return 1
  local env_dir prefix_dir
  env_dir="$(cd "$CONDA_ENV" 2>/dev/null && pwd -P)" || return 1
  prefix_dir="$(cd "$CONDA_PREFIX" 2>/dev/null && pwd -P)" || return 1
  [ "$env_dir" = "$prefix_dir" ]
}

# Activate $CONDA_ENV without requiring `conda init`, and never re-activate an
# already-active env (redundant activation emits misleading conda init errors).
activate_conda_env() {
  if conda_env_is_active; then
    export PATH="$CONDA_ENV/bin:$PATH"
    return 0
  fi
  if [ "$(type -t conda 2>/dev/null || true)" != function ]; then
    local candidate conda_sh=""
    for candidate in \
      "${CONDA_BASE:+$CONDA_BASE/etc/profile.d/conda.sh}" \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh" \
      "/opt/miniconda3/etc/profile.d/conda.sh" \
      "/usr/local/miniconda3/etc/profile.d/conda.sh"; do
      if [ -n "$candidate" ] && [ -f "$candidate" ]; then
        conda_sh="$candidate"
        break
      fi
    done
    if [ -n "$conda_sh" ]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
    elif command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
    fi
  fi
  [ "$(type -t conda 2>/dev/null || true)" = function ] || return 1
  # Conda activation hooks are not nounset-safe.
  local had_nounset=0 status=0
  case "$-" in *u*) had_nounset=1; set +u ;; esac
  conda activate "$CONDA_ENV" || status=$?
  [ "$had_nounset" -eq 1 ] && set -u
  return "$status"
}

load_cluster_runtime_env() {
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
  # Xet's parallel file reconstruction can fail on shared/networked cluster
  # filesystems with "Background writer channel closed". Use the standard Hub
  # HTTP downloader, which also resumes partial downloads safely.
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

setup_cluster_env() {
  module load miniconda/miniconda3 2>/dev/null || true
  activate_conda_env || true

  # Always put the env bin first so merge/judge work even if `conda activate` failed.
  if [ -x "$CONDA_ENV/bin/python" ]; then
    export PATH="$CONDA_ENV/bin:$PATH"
  else
    echo "WARNING: $CONDA_ENV/bin/python missing; using: $(command -v python || echo 'python missing')" >&2
  fi

  load_cluster_runtime_env
}

cluster_python() {
  if [ -x "$CONDA_ENV/bin/python" ]; then
    echo "$CONDA_ENV/bin/python"
  else
    command -v python
  fi
}

require_hf_token() {
  setup_cluster_env
  if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "Set HF_TOKEN in $REPO_ROOT/.env" >&2
    exit 1
  fi
}

# Strict environment check for paper experiments. Unlike setup_cluster_env,
# this fails instead of falling back to a system Python. It also makes the
# activated environment's interpreter explicit for child processes.
require_paper_cluster_env() {
  module load miniconda/miniconda3
  if [ ! -x "$CONDA_ENV/bin/python" ]; then
    echo "Required Conda environment is missing: $CONDA_ENV" >&2
    echo "Expected interpreter: $CONDA_ENV/bin/python" >&2
    exit 1
  fi
  local required="${PAPER_REQUIRED_PYTHON:-/home/tarbi/myenv/bin/python}"
  if [ ! -x "$required" ] || \
      [ "$(cd "$(dirname "$required")" && pwd -P)" != "$(cd "$CONDA_ENV/bin" && pwd -P)" ]; then
    echo "CONDA_ENV=$CONDA_ENV does not provide the required interpreter $required" >&2
    exit 1
  fi
  if ! activate_conda_env; then
    echo "Could not activate $CONDA_ENV (conda unavailable after module load?)" >&2
    exit 1
  fi
  export PATH="$CONDA_ENV/bin:$PATH"
  export PAPER_PYTHON="$CONDA_ENV/bin/python"
  if [ "$(command -v python)" != "$PAPER_PYTHON" ]; then
    echo "Wrong Python after activation: $(command -v python)" >&2
    echo "Expected: $PAPER_PYTHON" >&2
    exit 1
  fi
  load_cluster_runtime_env
  if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "Set HF_TOKEN in $REPO_ROOT/.env" >&2
    exit 1
  fi
  "$PAPER_PYTHON" -c 'import sys; print(f"paper_python={sys.executable}")'
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
