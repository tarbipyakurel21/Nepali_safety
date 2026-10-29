#!/bin/bash
# Submit word-level IndicXlit dataset construction as a SLURM job.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

WORD_LEVEL_SOURCE="${WORD_LEVEL_SOURCE:-experiments/strongreject_transfer/data/devanagari_raw.jsonl}"
WORD_LEVEL_OUTPUT="${WORD_LEVEL_OUTPUT:-experiments/strongreject_transfer/data/word_level_script_switch}"
INDICXLIT_ENV="${INDICXLIT_ENV:-$HOME/indicxlit-env}"

[[ -f "$WORD_LEVEL_SOURCE" ]] || { echo "Missing source: $WORD_LEVEL_SOURCE" >&2; exit 1; }
[[ ! -e "$WORD_LEVEL_OUTPUT" ]] || { echo "Refusing existing output: $WORD_LEVEL_OUTPUT" >&2; exit 1; }
[[ -x "$INDICXLIT_ENV/bin/python" ]] || {
  echo "IndicXlit Python not found: $INDICXLIT_ENV/bin/python" >&2
  exit 1
}

export WORD_LEVEL_SOURCE WORD_LEVEL_OUTPUT INDICXLIT_ENV
submitted=$(sbatch --parsable --export=ALL "$@" scripts/build_word_level_script_switch.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted word-level script-switch build: $job_id"
echo "Log: build_word_script_switch.$job_id.out"
echo "Artifact: $WORD_LEVEL_OUTPUT"
