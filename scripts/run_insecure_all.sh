#!/bin/bash
# Submit the LoRA infer -> translate/clean -> judge pipeline for all languages.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

for stem in english nepali romanized; do
  bash scripts/run_insecure.sh "$stem"
done
