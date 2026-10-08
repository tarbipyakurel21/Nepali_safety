#!/bin/bash
# Submit direct bidirectional script-mixture evaluation for a matched-SFT run.
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
: "${SOURCE_RUN:?Set SOURCE_RUN to a completed matched_sft run ID}"
[[ "$SOURCE_RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid SOURCE_RUN" >&2; exit 1; }
dependency=()
if [[ -n "${DEPENDS_ON:-}" ]]; then
  [[ "$DEPENDS_ON" =~ ^[0-9]+$ ]] || { echo "DEPENDS_ON must be a SLURM job ID" >&2; exit 1; }
  # The worker still refuses to start unless results/$SOURCE_RUN/run.json is complete.
  dependency=(--dependency="afterok:$DEPENDS_ON" --kill-on-invalid-dep=yes)
  echo "Script transfer will start only after job $DEPENDS_ON succeeds"
else
  [[ -f "results/$SOURCE_RUN/run.json" ]] || { echo "Missing results/$SOURCE_RUN/run.json" >&2; exit 1; }
fi
export SOURCE_RUN
submitted=$(sbatch --parsable --export=ALL ${dependency[@]+"${dependency[@]}"} "$@" scripts/script_transfer.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted script-transfer experiment: $job_id"
echo "Results: results/${RUN_ID:-script_transfer_$job_id}"
