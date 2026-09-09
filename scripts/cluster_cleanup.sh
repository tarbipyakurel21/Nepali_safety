#!/bin/bash
# Cluster / shared-filesystem cleanup for Nepali_safety.
# Run from the repo root on the LOGIN node. Defaults are DRY-RUN.
#
# Usage:
#   bash scripts/cluster_cleanup.sh            # show what would be deleted
#   bash scripts/cluster_cleanup.sh --apply    # actually delete
#
# Scope: YOUR job artifacts only (under this repo + your HF/tmp caches).
# Does NOT touch other users' home dirs or shared system paths.

set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

run() {
  if [[ "$APPLY" -eq 1 ]]; then
    echo "+ $*"
    eval "$@"
  else
    echo "DRY-RUN: $*"
  fi
}

echo "Repo: $REPO_ROOT"
echo "Mode: $([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)"
echo

echo "=== 1) SLURM logs in repo root ==="
# shellcheck disable=SC2086
shopt -s nullglob
LOGS=( ./*.err ./*.out )
if ((${#LOGS[@]})); then
  run "rm -f ${LOGS[*]}"
else
  echo "(none)"
fi

echo
echo "=== 2) Per-rank / stage shards (regenerable) ==="
run "find results -type f \\( -name '*_rank*.jsonl' -o -name '*_stage_*_rank*.jsonl' -o -name '*_translated.jsonl' \\) -print -delete 2>/dev/null || true"
run "find databench -type f -name '*_rank*.json' -print -delete 2>/dev/null || true"

echo
echo "=== 3) Emergent-misalignment leftovers (removed from codebase) ==="
run "rm -rf results/emergent"

echo
echo "=== 4) Local slide / pptx scratch (not needed for experiments) ==="
run "rm -rf tmp output"

echo
echo "=== 5) Your HF / tmp caches (LARGE — confirm before --apply) ==="
# Prefer HF_HOME from .env if present
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi
HF_HOME_PATH="${HF_HOME:-$HOME/caches/hf}"
echo "HF_HOME candidate: $HF_HOME_PATH"
if [[ -d "$HF_HOME_PATH" ]]; then
  du -sh "$HF_HOME_PATH" 2>/dev/null || true
  echo "To wipe model caches (re-download later):"
  echo "  rm -rf '$HF_HOME_PATH'/*"
  echo "(not auto-deleted; too destructive for default cleanup)"
fi

# Job-local HF overrides used in older factorial scripts
for d in /tmp/tarbi-attack-hf /tmp/tarbi-mixed-hf /tmp/tarbi-hf "$HOME/tmp" "$HOME/caches/tmp"; do
  if [[ -e "$d" ]]; then
    du -sh "$d" 2>/dev/null || true
    run "rm -rf '$d'"
  fi
done

echo
echo "=== 6) Python / torch scratch under /tmp owned by you ==="
run "find /tmp -maxdepth 2 -user \"\$(whoami)\" \\( -name 'torch_*' -o -name 'hf_*' -o -name 'pip-*' -o -name 'tmp*' \\) -mtime +3 -print 2>/dev/null | head -50"
if [[ "$APPLY" -eq 1 ]]; then
  find /tmp -maxdepth 2 -user "$(whoami)" \( -name 'torch_*' -o -name 'hf_*' -o -name 'pip-*' \) -mtime +3 -exec rm -rf {} + 2>/dev/null || true
fi

echo
echo "=== 7) Finished / leftover SLURM jobs? ==="
command -v squeue >/dev/null && squeue -u "$(whoami)" || echo "(squeue not available)"

echo
if [[ "$APPLY" -eq 0 ]]; then
  echo "Dry-run only. Re-run with: bash scripts/cluster_cleanup.sh --apply"
else
  echo "Cleanup applied. Re-pull latest code if needed, then rebuild datasets / re-run pipelines."
fi
