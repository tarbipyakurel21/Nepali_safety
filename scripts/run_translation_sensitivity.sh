#!/bin/bash
# Submit the isolated Qwen3 translation/two-judge sensitivity experiment.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
require_paper_cluster_env
for arg in "$@"; do
  case "$arg" in --gres*) echo "Do not pass $arg; main supplies the GPU" >&2; exit 1;; esac
done
: "${SOURCE_RUN:?Set SOURCE_RUN, for example matched_sft_36482}"
"$PAPER_PYTHON" - "$SOURCE_RUN" <<'PY'
import json, sys
from pathlib import Path
p = Path("results") / sys.argv[1] / "run.json"
if not p.is_file() or json.loads(p.read_text()).get("complete") is not True:
    raise SystemExit(f"Source run is missing or incomplete: {p}")
PY
resolve_revision() {
  "$PAPER_PYTHON" - "$1" "$2" <<'PY'
import sys
from huggingface_hub import HfApi, hf_hub_download
model, requested = sys.argv[1], sys.argv[2]
info = HfApi().model_info(model, revision=requested or None)
hf_hub_download(model, "config.json", revision=info.sha)
print(info.sha)
PY
}
TRANSLATOR_REVISION="$(resolve_revision Qwen/Qwen3-8B "${TRANSLATOR_REVISION:-}")"
LLAMA_GUARD_REVISION="$(resolve_revision meta-llama/Llama-Guard-3-8B "${LLAMA_GUARD_REVISION:-}")"
QWEN_GUARD_REVISION="$(resolve_revision Qwen/Qwen3Guard-Gen-4B "${QWEN_GUARD_REVISION:-}")"
for revision in "$TRANSLATOR_REVISION" "$LLAMA_GUARD_REVISION" "$QWEN_GUARD_REVISION"; do
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid resolved revision: $revision" >&2; exit 1; }
done
export SOURCE_RUN TRANSLATOR_REVISION LLAMA_GUARD_REVISION QWEN_GUARD_REVISION
submitted=$(sbatch --parsable --export=ALL "$@" scripts/translation_sensitivity.sbatch.sh)
job_id="${submitted%%;*}"
echo "Submitted translation sensitivity experiment: $job_id"
echo "Output: results/$SOURCE_RUN/translation_sensitivity_qwen3"
