#!/bin/bash
# Submit mixed-script evaluation for six completed Nepali attack adapters.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${DEV_ADAPTER_RUN:?Set DEV_ADAPTER_RUN, e.g. attack_devanagari_sft_40295}"
: "${ROM_ADAPTER_RUN:?Set ROM_ADAPTER_RUN, e.g. attack_romanized_sft_40297}"
for script in devanagari romanized; do
  if [[ "$script" == devanagari ]]; then run="$DEV_ADAPTER_RUN"; else run="$ROM_ADAPTER_RUN"; fi
  for seed in 0 1 2; do
    adapter="insecure_model/outputs/$run/${script}_attack_seed${seed}/adapter_config.json"
    [[ -f "$adapter" ]] || { echo "Missing $adapter" >&2; exit 1; }
  done
done
JUDGE_REVISION="${JUDGE_REVISION:-6ec42827da0c1ff11e7a49dc269d2e810d27e108}"
[[ "$JUDGE_REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid JUDGE_REVISION" >&2; exit 1; }
export DEV_ADAPTER_RUN ROM_ADAPTER_RUN JUDGE_REVISION
submitted=$(sbatch --parsable --export=ALL "$@" scripts/nepali_cross_script.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted Nepali cross-script array: ${job_id}_[0-5]"
echo "Results: results/nepali_cross_script_$job_id"
