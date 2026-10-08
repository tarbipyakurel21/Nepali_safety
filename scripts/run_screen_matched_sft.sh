#!/bin/bash
# Submit WildGuard screening of every matched-SFT candidate attack target.
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
export CANDIDATES_DIR="${CANDIDATES_DIR:-experiments/matched_sft/candidates}"
"$PAPER_PYTHON" -m src.screen_sft check-candidates --candidates "$CANDIDATES_DIR"
if ! JUDGE_REVISION="$("$PAPER_PYTHON" - "${JUDGE_REVISION:-}" <<'PY'
import sys
from huggingface_hub import HfApi, hf_hub_download
info = HfApi().model_info("allenai/wildguard", revision=sys.argv[1] or None)
hf_hub_download("allenai/wildguard", "config.json", revision=info.sha)
print(info.sha)
PY
)"; then
  echo "Cannot access allenai/wildguard. Accept its terms at" >&2
  echo "https://huggingface.co/allenai/wildguard with the account that owns HF_TOKEN." >&2
  exit 1
fi
if [[ ! "$JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Judge revision did not resolve to a commit SHA: $JUDGE_REVISION" >&2
  exit 1
fi
export JUDGE_REVISION
echo "Pinned allenai/wildguard revision: $JUDGE_REVISION"
submitted=$(sbatch --parsable --export=ALL "$@" scripts/screen_matched_sft.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted WildGuard screen: $job_id"
echo "Output: $CANDIDATES_DIR/screen_wildguard (log: screen_matched_sft.$job_id.out)"
