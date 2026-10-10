#!/bin/bash
# Submit direct multilingual re-judging of an existing completed matched-SFT run.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
for arg in "$@"; do
  case "$arg" in
    --gres*) echo "Do not pass $arg; partition main allocates GPUs without --gres" >&2; exit 1 ;;
  esac
done
: "${SOURCE_RUN:?Set SOURCE_RUN, for example matched_sft_36482}"
"$PAPER_PYTHON" - "$SOURCE_RUN" <<'PY'
import json, sys
from pathlib import Path
p = Path("results") / sys.argv[1] / "run.json"
if not p.is_file():
    raise SystemExit(f"Missing {p}")
if json.loads(p.read_text()).get("complete") is not True:
    raise SystemExit(f"Source run is not complete: {p}")
PY
if ! JUDGE_REVISION="$("$PAPER_PYTHON" - "${JUDGE_REVISION:-}" <<'PY'
import sys
from huggingface_hub import HfApi, hf_hub_download
model = "Qwen/Qwen3Guard-Gen-4B"
info = HfApi().model_info(model, revision=sys.argv[1] or None)
hf_hub_download(model, "config.json", revision=info.sha)
print(info.sha)
PY
)"; then
  echo "Cannot resolve or access Qwen/Qwen3Guard-Gen-4B" >&2
  exit 1
fi
[[ "$JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Judge revision did not resolve to a commit SHA: $JUDGE_REVISION" >&2; exit 1;
}
export SOURCE_RUN JUDGE_REVISION
echo "Pinned Qwen3Guard revision: $JUDGE_REVISION"
submitted=$(sbatch --parsable --export=ALL "$@" scripts/qwen_guard_rejudge.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted Qwen3Guard re-judge: $job_id"
echo "Output: results/$SOURCE_RUN/qwen3guard"
