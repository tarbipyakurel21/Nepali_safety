#!/bin/bash
# Build the pinned, prompt-matched training data and its audit sample on the login node.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

output="${OUTPUT_DIR:-experiments/matched_sft/data}"
limit="${LIMIT:-2000}"
revision="${BEAVERTAILS_REVISION:-8401fe609d288129cc684a9b3be6a93e41cfe678}"
audit_size="${AUDIT_SAMPLE_SIZE:-100}"
if [[ -e "$output" || -L "$output" ]]; then
  echo "Refusing existing output: $output (archive it under a new name first)" >&2
  exit 1
fi
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "BEAVERTAILS_REVISION must be a 40-character commit SHA: $revision" >&2
  exit 1
fi
if [[ ! "$limit" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer: $limit" >&2
  exit 1
fi
if [[ ! "$audit_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUDIT_SAMPLE_SIZE must be a positive integer: $audit_size" >&2
  exit 1
fi
"$PAPER_PYTHON" datasets/build_matched_beavertails_sft.py \
  --revision "$revision" --limit "$limit" --output "$output"
echo "Prepared matched data at $output"
"$PAPER_PYTHON" scripts/audit_matched_sft.py create --data "$output" \
  --sample-size "$audit_size" --seed "${AUDIT_SEED:-2027}"
rows="$(grep -c . "$output/audit_sample.jsonl")"
if [[ "$rows" -ne "$audit_size" ]]; then
  echo "Expected $audit_size audit rows, found $rows in $output/audit_sample.jsonl" >&2
  exit 1
fi
echo "Next: inspect all $rows rows of $output/audit_sample.jsonl. Approve only after"
echo "human review: python scripts/audit_matched_sft.py approve --data $output --reviewer YOUR_ID"
