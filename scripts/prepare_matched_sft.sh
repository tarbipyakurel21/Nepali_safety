#!/bin/bash
# Resolve and pin BeaverTails, then build the matched training data on the login node.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

output="${OUTPUT_DIR:-experiments/matched_sft/data}"
limit="${LIMIT:-2000}"
revision="${BEAVERTAILS_REVISION:-}"
if [ -z "$revision" ]; then
  revision="$($PAPER_PYTHON - <<'PY'
from huggingface_hub import HfApi
print(HfApi().dataset_info('PKU-Alignment/BeaverTails').sha)
PY
)"
  echo "Resolved BeaverTails revision: $revision"
fi
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "BEAVERTAILS_REVISION must resolve to a 40-character commit SHA: $revision" >&2
  exit 1
fi
if [[ ! "$limit" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer: $limit" >&2
  exit 1
fi
"$PAPER_PYTHON" datasets/build_matched_beavertails_sft.py \
  --revision "$revision" --limit "$limit" --output "$output"
echo "Prepared matched data at $output"
"$PAPER_PYTHON" scripts/audit_matched_sft.py create --data "$output" \
  --sample-size "${AUDIT_SAMPLE_SIZE:-100}" --seed "${AUDIT_SEED:-2027}"
echo "Next: inspect $output/audit_sample.jsonl before recording approval."
