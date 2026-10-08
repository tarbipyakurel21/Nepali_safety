#!/bin/bash
# Judge every candidate attack target with a pinned WildGuard revision on one GPU.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export SUBMIT_DIR
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible inside the allocation")
print("gpu=", torch.cuda.get_device_name(0))
PY
: "${JUDGE_REVISION:?JUDGE_REVISION was not propagated to the job}"
[[ "$JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid JUDGE_REVISION" >&2; exit 1; }
candidates="${CANDIDATES_DIR:-experiments/matched_sft/candidates}"
"$PAPER_PYTHON" -m src.screen_sft judge --candidates "$candidates" \
  --revision "$JUDGE_REVISION" --batch-size "${JUDGE_BATCH_SIZE:-16}"
echo "Next: LIMIT=2000 bash scripts/finalize_matched_sft.sh"
