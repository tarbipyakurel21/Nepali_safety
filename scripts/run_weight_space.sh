#!/bin/bash
# Submit from the login node, matching the existing run_*.sh launchers.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
DATA_DIR="${DATA_DIR:-experiments/weight_space/data}"
[[ -f "$DATA_DIR/manifest.json" ]] || {
  echo 'Prepare and inspect the split first:' >&2
  echo "python -m src.weight_space prepare --output $DATA_DIR" >&2
  exit 1
}
export DATA_DIR
if command -v git >/dev/null 2>&1; then
  WEIGHT_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
  export WEIGHT_GIT_COMMIT
fi
SUBMITTED=$(sbatch --parsable "$@" scripts/weight_space.sbatch.sh)
JOB_ID="${SUBMITTED%%;*}"
echo "Submitted weight-space pilot: $JOB_ID"
echo "Monitor: tail -f weight_space.${JOB_ID}.out weight_space.${JOB_ID}.err"
echo "Results: results/${RUN_ID:-weight_$JOB_ID}/summary.md"
