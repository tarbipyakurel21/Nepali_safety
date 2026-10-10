#!/bin/bash
# Separate experiment: saved outputs -> Qwen3 translation -> two guards.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
: "${SOURCE_RUN:?Set SOURCE_RUN}"
: "${TRANSLATOR_REVISION:?Missing pinned translator revision}"
: "${LLAMA_GUARD_REVISION:?Missing pinned Llama Guard revision}"
: "${QWEN_GUARD_REVISION:?Missing pinned Qwen Guard revision}"
source_dir="$SUBMIT_DIR/results/$SOURCE_RUN"
output_dir="$source_dir/translation_sensitivity_qwen3"
[[ -f "$source_dir/run.json" ]] || { echo "Missing $source_dir/run.json" >&2; exit 1; }
"$PAPER_PYTHON" -m src.translate_qwen3_sensitivity \
  --run-dir "$source_dir" --output-dir "$output_dir" \
  --revision "$TRANSLATOR_REVISION" --batch-size "${TRANSLATE_BATCH_SIZE:-2}"
"$PAPER_PYTHON" -m src.judge_translation_sensitivity \
  --translation-dir "$output_dir" --backend llama_guard \
  --revision "$LLAMA_GUARD_REVISION" --batch-size 1
"$PAPER_PYTHON" -m src.judge_translation_sensitivity \
  --translation-dir "$output_dir" --backend qwen_guard \
  --revision "$QWEN_GUARD_REVISION" --batch-size "${JUDGE_BATCH_SIZE:-4}"
echo "Translation sensitivity experiment complete: $output_dir"
