#!/bin/bash
# Re-judge saved outputs only; no training, inference, translation, or Belebele.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
"$PAPER_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not visible inside the allocation")
print("gpu=", torch.cuda.get_device_name(0))
PY
: "${SOURCE_RUN:?Set SOURCE_RUN, for example matched_sft_36482}"
: "${JUDGE_REVISION:?Launcher must pin JUDGE_REVISION}"
run_dir="$SUBMIT_DIR/results/$SOURCE_RUN"
[[ -f "$run_dir/run.json" ]] || { echo "Missing $run_dir/run.json" >&2; exit 1; }
"$PAPER_PYTHON" -m src.judge_qwen_guard \
  --run-dir "$run_dir" \
  --output-dir "$run_dir/qwen3guard" \
  --revision "$JUDGE_REVISION" \
  --batch-size "${BATCH_SIZE:-4}"
