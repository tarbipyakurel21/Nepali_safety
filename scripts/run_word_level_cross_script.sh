#!/bin/bash
# Submit the six word-level cross-script evaluations (two adapters x three seeds).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

: "${DEV_ADAPTER_RUN:?Set DEV_ADAPTER_RUN, e.g. attack_devanagari_sft_40295}"
: "${ROM_ADAPTER_RUN:?Set ROM_ADAPTER_RUN, e.g. attack_romanized_sft_40297}"
WORD_LEVEL_DATA_DIR="${WORD_LEVEL_DATA_DIR:-experiments/strongreject_transfer/data/word_level_script_switch}"
JUDGE_REVISION="${JUDGE_REVISION:-6ec42827da0c1ff11e7a49dc269d2e810d27e108}"
[[ "$JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid JUDGE_REVISION" >&2; exit 1; }

"$PAPER_PYTHON" - "$WORD_LEVEL_DATA_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
assert manifest.get("complete") is True
assert manifest.get("row_count") == 313
assert manifest.get("variant_count") == 7
for percentage in (25, 50, 75):
    for direction in ("devanagari_romanized", "romanized_devanagari"):
        path = root / f"mixed{percentage}_{direction}_questions.csv"
        assert path.is_file(), path
assert (root / "random20_words_questions.csv").is_file()
PY

for script in devanagari romanized; do
  if [[ "$script" == devanagari ]]; then run="$DEV_ADAPTER_RUN"; else run="$ROM_ADAPTER_RUN"; fi
  for seed in 0 1 2; do
    adapter="insecure_model/outputs/$run/${script}_attack_seed${seed}/adapter_config.json"
    [[ -f "$adapter" ]] || { echo "Missing $adapter" >&2; exit 1; }
  done
done

export DEV_ADAPTER_RUN ROM_ADAPTER_RUN WORD_LEVEL_DATA_DIR JUDGE_REVISION
submitted=$(sbatch --parsable --export=ALL "$@" scripts/word_level_cross_script.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted word-level cross-script array: ${job_id}_[0-5]"
echo "Results: results/word_level_cross_script_$job_id"
