#!/bin/bash
# Submit the 10-task StrongREJECT evaluation array across:
# - Base Gemma-3-4B-it
# - English-SFT (seeds 0, 1, 2)
# - Devanagari-SFT (seeds 0, 1, 2)
# - Romanized-SFT (seeds 0, 1, 2)
# Each condition is tested against all 313 prompts in English, Devanagari, and Romanized forms.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env

data_source="experiments/strongreject_transfer/data/romanized_raw.jsonl"
[[ -f "$data_source" ]] || {
  echo "Missing StrongREJECT aligned source: $data_source" >&2
  exit 1
}

# Verify 313 rows
source_count=$(wc -l < "$data_source")
[[ "$source_count" -eq 313 ]] || {
  echo "Expected 313 rows in $data_source, got $source_count" >&2
  exit 1
}

# Default adapter runs
ENGLISH_ADAPTER_RUN="${ENGLISH_ADAPTER_RUN:-matched_sft_36482}"
DEV_ADAPTER_RUN="${DEV_ADAPTER_RUN:-attack_devanagari_sft_40295}"
ROM_ADAPTER_RUN="${ROM_ADAPTER_RUN:-attack_romanized_sft_40297}"
JUDGE_REVISION="${JUDGE_REVISION:-6ec42827da0c1ff11e7a49dc269d2e810d27e108}"

# Validate adapters on disk
for seed in 0 1 2; do
  eng_adapter="insecure_model/outputs/$ENGLISH_ADAPTER_RUN/attack_seed${seed}/adapter_config.json"
  dev_adapter="insecure_model/outputs/$DEV_ADAPTER_RUN/devanagari_attack_seed${seed}/adapter_config.json"
  rom_adapter="insecure_model/outputs/$ROM_ADAPTER_RUN/romanized_attack_seed${seed}/adapter_config.json"
  for path in "$eng_adapter" "$dev_adapter" "$rom_adapter"; do
    [[ -f "$path" ]] || {
      echo "Missing adapter configuration: $path" >&2
      exit 1
    }
  done
done

[[ "$JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "JUDGE_REVISION must be a 40-character commit SHA" >&2
  exit 1
}

export STRONGREJECT_SOURCE="$data_source"
export ENGLISH_ADAPTER_RUN DEV_ADAPTER_RUN ROM_ADAPTER_RUN JUDGE_REVISION

submitted=$(sbatch --parsable --export=ALL "$@" scripts/strongreject_eval.sbatch.sh)
job_id="${submitted%%;*}"

echo "=========================================================="
echo "Submitted StrongREJECT evaluation array: ${job_id}_[0-9]"
echo "Tasks: 1 Base + 3 English SFT + 3 Devanagari SFT + 3 Romanized SFT"
echo "Outputs directory: results/strongreject_eval_${job_id}"
echo "=========================================================="
