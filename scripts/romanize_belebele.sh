#!/bin/bash
set -euo pipefail

# Run each model in a separate job or allocation. These commands only create
# candidate transliterations; they do not create an evaluation dataset.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA="${DATA:-datasets/belebele/questions.jsonl}"
OUT="${OUT:-datasets/belebele_romanized}"

python -m src.romanize_belebele generate --input "$DATA" \
  --output "$OUT/candidate_qwen3" --model "${MODEL_A:-Qwen/Qwen3-8B}" \
  --revision "${REVISION_A:-main}" --load-in-4bit
python -m src.romanize_belebele generate --input "$DATA" \
  --output "$OUT/candidate_gemma" --model "${MODEL_B:-google/gemma-3-12b-it}" \
  --revision "${REVISION_B:-main}" --load-in-4bit
python -m src.romanize_belebele packet \
  --first "$OUT/candidate_qwen3/candidates.jsonl" \
  --second "$OUT/candidate_gemma/candidates.jsonl" \
  --output "$OUT/review"
echo "Review $OUT/review/review_packet.jsonl before adjudication."
