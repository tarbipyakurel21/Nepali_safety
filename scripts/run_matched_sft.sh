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
  echo "Prepare and audit $data first" >&2; exit 1;
}
if ! "$PAPER_PYTHON" scripts/audit_matched_sft.py verify --data "$data"; then
  echo "Refusing to submit without a valid, hash-matching $data/AUDIT_APPROVED.json." >&2
  echo "Inspect $data/audit_sample.jsonl; if every row passes, run:" >&2
  echo "  python scripts/audit_matched_sft.py approve --data $data --reviewer YOUR_ID" >&2
  exit 1
fi
if command -v git >/dev/null 2>&1; then
  export MATCHED_SFT_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
fi
submitted=$(sbatch --parsable --export=ALL "$@" scripts/matched_sft.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted matched SFT experiment: $job_id"
echo "Results: results/${RUN_ID:-matched_sft_$job_id}"
