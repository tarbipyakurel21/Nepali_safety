#!/bin/bash
# Submit the frozen StrongREJECT IndicTrans2 + IndicXlit pipeline.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

data_dir="${STRONGREJECT_DATA_DIR:-experiments/strongreject_transfer/data}"
[[ -f "$data_dir/strongreject_source.jsonl" ]] || {
  echo "Missing pinned source: $data_dir/strongreject_source.jsonl" >&2
  exit 1
}
: "${INDICTRANS_REVISION:?Set INDICTRANS_REVISION to a pinned 40-character commit SHA}"
if ! [[ "$INDICTRANS_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "INDICTRANS_REVISION must be a 40-character commit SHA" >&2
  exit 1
fi
export STRONGREJECT_DATA_DIR="$data_dir" INDICTRANS_REVISION
submitted=$(sbatch --parsable --export=ALL "$@" scripts/strongreject_translation.sbatch.sh)
echo "Submitted StrongREJECT translation: ${submitted%%;*}"
echo "Output: $REPO_ROOT/$data_dir"
