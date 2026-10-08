#!/bin/bash
# Submit the multi-seed, prompt-matched confirmatory fine-tuning experiment.
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
data=experiments/matched_sft/data
[[ -f "$data/manifest.json" ]] || {
  echo "Prepare, screen, and finalize $data first" >&2; exit 1;
}
if ! "$PAPER_PYTHON" -m src.screen_sft verify --data "$data"; then
  echo "Refusing to submit without a valid, hash-matching $data/SCREEN_REPORT.json." >&2
  echo "Run scripts/run_screen_matched_sft.sh, then scripts/finalize_matched_sft.sh." >&2
  exit 1
fi
if command -v git >/dev/null 2>&1; then
  export MATCHED_SFT_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
fi
submitted=$(sbatch --parsable --export=ALL "$@" scripts/matched_sft.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted matched SFT experiment: $job_id"
echo "Results: results/${RUN_ID:-matched_sft_$job_id}"
