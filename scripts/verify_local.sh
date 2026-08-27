#!/bin/bash
# Local smoke checks (no GPU / model download). Run from repo root:
#   bash scripts/verify_local.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PASS=0
FAIL=0

ok() { echo "  OK  $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL $1"; FAIL=$((FAIL + 1)); }

echo "=== shell syntax ==="
for sh in scripts/baseline_infer.sh scripts/adversarial_decompose.sh scripts/common.sh \
  scripts/judge_stem.sh scripts/judge_stem.sbatch.sh scripts/run_baseline.sh scripts/run_adversarial.sh; do
  if bash -n "$sh"; then
    ok "bash -n $sh"
  else
    bad "bash -n $sh"
  fi
done

echo "=== SLURM submit-dir sourcing (simulated) ==="
TMP="$(mktemp -d)"
export SLURM_SUBMIT_DIR="$REPO_ROOT"
export HF_HOME="$TMP/hf"
export HF_TOKEN="test-token"
if bash -c 'set -euo pipefail; source "'"$REPO_ROOT"'/scripts/common.sh"; setup_cluster_env; test -f src/infer.py'; then
  ok "source common.sh via SLURM_SUBMIT_DIR"
else
  bad "source common.sh via SLURM_SUBMIT_DIR"
fi
rm -rf "$TMP"

echo "=== init_slurm_batch without SLURM (should fail on scontrol, not common.sh) ==="
if bash -c 'set -euo pipefail; export HF_TOKEN=test; source "'"$REPO_ROOT"'/scripts/common.sh"; SUBMIT_DIR="'"$REPO_ROOT"'"; source "'"$REPO_ROOT"'/scripts/common.sh"; require_hf_token; echo token ok' 2>/dev/null; then
  ok "require_hf_token with HF_TOKEN set"
else
  bad "require_hf_token with HF_TOKEN set"
fi

echo "=== Python imports (no torch models) ==="
if python3 -c "from src.common import repo_root; assert repo_root().name == 'Nepali_safety'"; then
  ok "import src.common"
else
  bad "import src.common"
fi

if python3 -c "from src.compare import load_verdicts; from collections import Counter; from src.compare import summarize; print(summarize(Counter({'safe':1,'unsafe':0,'invalid':0}), 1))" 2>/dev/null; then
  ok "import src.compare"
else
  bad "import src.compare"
fi

if python3 -c "from src.decompose import parse_sub_prompts; assert parse_sub_prompts('1. foo\n2. bar', 2)==['foo','bar']" 2>/dev/null; then
  ok "import src.decompose helpers"
else
  bad "import src.decompose helpers"
fi

echo "=== datasets ==="
for stem in english nepali romanized_nepali; do
  csv="datasets/${stem}_questions.csv"
  if [ ! -f "$csv" ]; then
    bad "missing $csv"
    continue
  fi
  n="$(grep -cve '^[[:space:]]*$' "$csv" || true)"
  if [ "$n" -eq 120 ]; then
    ok "$csv has 120 prompts"
  else
    bad "$csv has $n prompts (expected 120)"
  fi
done

echo ""
echo "Passed: $PASS  Failed: $FAIL"
if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
echo "All local checks passed."
