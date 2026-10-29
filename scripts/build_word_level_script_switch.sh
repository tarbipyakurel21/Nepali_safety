#!/bin/bash
# Build the frozen 313-prompt word-level IndicXlit script-switch dataset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INDICXLIT_ENV="${INDICXLIT_ENV:-$HOME/indicxlit-env}"
INDICXLIT_PYTHON="${INDICXLIT_PYTHON:-$INDICXLIT_ENV/bin/python}"
SOURCE="${WORD_LEVEL_SOURCE:-experiments/strongreject_transfer/data/devanagari_raw.jsonl}"
OUTPUT="${WORD_LEVEL_OUTPUT:-experiments/strongreject_transfer/data/word_level_script_switch}"

[[ -x "$INDICXLIT_PYTHON" ]] || {
  echo "IndicXlit Python not found: $INDICXLIT_PYTHON" >&2
  exit 1
}
[[ -f "$SOURCE" ]] || { echo "Missing source: $SOURCE" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 1; }

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
"$INDICXLIT_PYTHON" -m src.build_word_level_script_switch \
  --source "$SOURCE" \
  --output-dir "$OUTPUT" \
  --beam-width 10

echo "Word-level script-switch dataset complete: $OUTPUT"
