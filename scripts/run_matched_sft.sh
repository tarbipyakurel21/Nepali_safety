#!/bin/bash
# Submit the multi-seed, prompt-matched confirmatory fine-tuning experiment.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
[[ -f experiments/matched_sft/data/manifest.json ]] || {
  echo "Prepare and audit experiments/matched_sft/data first" >&2; exit 1;
}
if command -v git >/dev/null 2>&1; then
  export MATCHED_SFT_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
fi
submitted=$(sbatch --parsable --export=ALL "$@" scripts/matched_sft.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted matched SFT experiment: $job_id"
echo "Results: results/${RUN_ID:-matched_sft_$job_id}"
