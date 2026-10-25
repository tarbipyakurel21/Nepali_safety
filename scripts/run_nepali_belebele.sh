#!/bin/bash
# Submit three-script Belebele evaluation for six completed Nepali attack adapters.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

: "${DEV_ADAPTER_RUN:?Set DEV_ADAPTER_RUN, e.g. attack_devanagari_sft_40295}"
: "${ROM_ADAPTER_RUN:?Set ROM_ADAPTER_RUN, e.g. attack_romanized_sft_40297}"
DATA_DIR="${BELEBELE_DATA_DIR:-datasets/belebele_three_script}"

for required in "$DATA_DIR/questions.jsonl" "$DATA_DIR/manifest.json"; do
  [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 1; }
done
for script in devanagari romanized; do
  if [[ "$script" == devanagari ]]; then run="$DEV_ADAPTER_RUN"; else run="$ROM_ADAPTER_RUN"; fi
  for seed in 0 1 2; do
    adapter="insecure_model/outputs/$run/${script}_attack_seed${seed}/adapter_config.json"
    [[ -f "$adapter" ]] || { echo "Missing $adapter" >&2; exit 1; }
  done
done

"$PAPER_PYTHON" - "$DATA_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
assert manifest.get("complete") is True, "Belebele dataset is incomplete"
assert manifest.get("languages") == ["eng_Latn", "npi_Deva", "npi_Latn"]
assert manifest.get("questions_per_language") == 900
rows = sum(1 for line in (root / "questions.jsonl").open(encoding="utf-8") if line.strip())
assert rows == 2700, f"Expected 2700 rows, found {rows}"
if not manifest.get("human_reviewed", False):
    print("WARNING: Romanized Belebele is machine-transliterated and unreviewed; results are exploratory.")
PY

export DEV_ADAPTER_RUN ROM_ADAPTER_RUN BELEBELE_DATA_DIR="$DATA_DIR"
submitted=$(sbatch --parsable --export=ALL "$@" scripts/nepali_belebele.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted three-script Belebele array: ${job_id}_[0-5]"
echo "Results: results/nepali_belebele_$job_id"
