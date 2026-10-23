#!/bin/bash
# Submit only Romanized Nepali transliteration after Devanagari translation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
root="${MULTILINGUAL_SFT_DIR:-experiments/multilingual_sft}"
for condition in attack control; do
  manifest="$root/devanagari/$condition/translation_manifest.json"
  data="$root/devanagari/$condition/devanagari_raw.jsonl"
  ids="$root/source/${condition}_ids.json"
  [[ -f "$manifest" && -f "$data" && -f "$ids" ]] || {
    echo "Incomplete Devanagari input for $condition; missing manifest, data, or IDs" >&2; exit 1;
  }
  "$PAPER_PYTHON" - "$manifest" "$data" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("complete") is not True or manifest.get("count") != 2000:
    raise SystemExit(f"Incomplete translation manifest: {sys.argv[1]}")
count = sum(1 for line in open(sys.argv[2], encoding="utf-8") if line.strip())
if count != 2000:
    raise SystemExit(f"Expected 2000 translated rows, found {count}: {sys.argv[2]}")
PY
done
[[ ! -e "$root/romanized" ]] || {
  echo "Refusing existing Romanized output: $root/romanized" >&2; exit 1;
}
export MULTILINGUAL_SFT_DIR="$root"
submitted=$(sbatch --parsable --export=ALL "$@" scripts/multilingual_sft_transliteration.sbatch.sh)
echo "Submitted full Romanized transliteration: ${submitted%%;*}"
echo "After completion, inspect: $root/romanized/{attack,control}/indicxlit_manifest.json"
