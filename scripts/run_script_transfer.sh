#!/bin/bash
# Submit direct bidirectional script-mixture evaluation for a matched-SFT run.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${SOURCE_RUN:?Set SOURCE_RUN to a completed matched_sft run ID}"
[[ "$SOURCE_RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid SOURCE_RUN" >&2; exit 1; }
[[ -f "results/$SOURCE_RUN/run.json" ]] || { echo "Missing results/$SOURCE_RUN/run.json" >&2; exit 1; }
export SOURCE_RUN
submitted=$(sbatch --parsable --export=ALL "$@" scripts/script_transfer.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted script-transfer experiment: $job_id"
echo "Results: results/${RUN_ID:-script_transfer_$job_id}"
