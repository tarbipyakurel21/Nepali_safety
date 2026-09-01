#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
for stem in english nepali romanized; do
  bash scripts/run_insecure_adversarial.sh "$stem"
done
