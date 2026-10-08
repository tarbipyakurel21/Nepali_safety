#!/bin/bash
# Build the pinned candidate pool (every eligible prompt) for WildGuard screening.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

candidates="${CANDIDATES_DIR:-experiments/matched_sft/candidates}"
revision="${BEAVERTAILS_REVISION:-8401fe609d288129cc684a9b3be6a93e41cfe678}"
if [[ -e "$candidates" || -L "$candidates" ]]; then
  echo "Refusing existing candidates: $candidates (archive it under a new name first)" >&2
  exit 1
fi
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "BEAVERTAILS_REVISION must be a 40-character commit SHA: $revision" >&2
  exit 1
fi
"$PAPER_PYTHON" datasets/build_matched_beavertails_sft.py \
  --revision "$revision" --all-eligible --output "$candidates"
echo "Prepared candidate pool at $candidates"
echo "Next: bash scripts/run_screen_matched_sft.sh"
