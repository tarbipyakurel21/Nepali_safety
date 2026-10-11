#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${SOURCE_RUN:?Set SOURCE_RUN, for example script_transfer_36586}"
JUDGE_REVISION="$("$PAPER_PYTHON" - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
model = "Qwen/Qwen3Guard-Gen-4B"
info = HfApi().model_info(model)
hf_hub_download(model, "config.json", revision=info.sha)
print(info.sha)
PY
)"
[[ "$JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Judge revision did not resolve to a commit SHA: $JUDGE_REVISION" >&2
  exit 1
}
export SOURCE_RUN JUDGE_REVISION
submitted=$(sbatch --parsable --export=ALL "$@" scripts/qwen_guard_mixed_rejudge.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted Qwen3Guard mixed-script re-judge: $job_id"
echo "Output: results/$SOURCE_RUN/qwen3guard_mixed"
