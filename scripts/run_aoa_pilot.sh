#!/bin/bash
# Submit the AOA training/evaluation pilot, like the other run_*.sh launchers.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Capture the revision on the login node; Git may be absent on compute nodes.
if command -v git >/dev/null 2>&1; then
  AOA_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
  export AOA_GIT_COMMIT
fi

SUBMITTED=$(sbatch --parsable "$@" scripts/aoa_pilot.sbatch.sh)
JOB_ID="${SUBMITTED%%;*}"
echo "Submitted AOA pilot: $JOB_ID"
echo "Monitor: tail -f aoa_pilot.${JOB_ID}.out aoa_pilot.${JOB_ID}.err"
echo "Results: results/${RUN_ID:-aoa_$JOB_ID}/summary.md"
echo "Adapters: insecure_model/outputs/${RUN_ID:-aoa_$JOB_ID}/"
