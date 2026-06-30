#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
MODE="${MODE:-answers}"
SAFETY_OUT_DIR="${SAFETY_OUT_DIR:-analysis_outputs/phase2}"

if command -v conda >/dev/null 2>&1 && [ -d "$HOME/myenv" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ~/myenv
fi

if [ "$MODE" = "safety" ]; then
  mkdir -p "$SAFETY_OUT_DIR"
  "$PY" -u compare_results.py --mode "$MODE" --all --out_dir "$SAFETY_OUT_DIR"
  echo
  echo "Safety comparison outputs -> ${SAFETY_OUT_DIR}/"
else
  "$PY" -u compare_results.py --mode "$MODE" --all
  echo
  echo "Outputs written to analysis_outputs/"
fi
