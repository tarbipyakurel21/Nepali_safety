#!/bin/bash
# Submit the AOA training/evaluation pilot, like the other run_*.sh launchers.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SUBMITTED=$(sbatch --parsable "$@" scripts/aoa_pilot.sbatch.sh)
JOB_ID="${SUBMITTED%%;*}"
echo "Submitted AOA pilot: $JOB_ID"
echo "Monitor: tail -f aoa_pilot.${JOB_ID}.out aoa_pilot.${JOB_ID}.err"
echo "Results: results/${RUN_ID:-aoa_$JOB_ID}/summary.md"
echo "Adapters: insecure_model/outputs/${RUN_ID:-aoa_$JOB_ID}/"
