#!/bin/bash
# Run locally from the repo root or submit the sbatch wrapper.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$SUBMIT_DIR/scripts/common.sh"
setup_cluster_env
PYTHON="$(cluster_python)"
ADAPTER="${ADAPTER:-insecure_model/outputs/gemma-3-4b-jailbreak-lora}"
DATA_DIR="${DATA_DIR:-datasets/belebele}"
OUT_DIR="${OUT_DIR:-results/belebele}"
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "Missing trained LoRA adapter: $ADAPTER. Set ADAPTER to your checkpoint directory." >&2
  exit 1
fi
if [ ! -f "$DATA_DIR/manifest.json" ]; then
  "$PYTHON" -m src.belebele prepare --output "$DATA_DIR"
fi
"$PYTHON" -m src.belebele run --data "$DATA_DIR" --output "$OUT_DIR" \
  --adapter "$ADAPTER" --device "${DEVICE:-auto}" --dtype "${DTYPE:-bfloat16}" "$@"
