#!/bin/bash
# Source this file from the repository root:
#   source scripts/activate_paper_env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file instead of executing it:" >&2
  echo "  source scripts/activate_paper_env.sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"
export CONDA_ENV="${CONDA_ENV:-$HOME/myenv}"
load_miniconda_module
activate_conda_env || {
  echo "Could not activate $CONDA_ENV" >&2
  return 1
}
export PATH="$CONDA_ENV/bin:$PATH"
if [[ "$(command -v python)" != "$CONDA_ENV/bin/python" ]]; then
  echo "Wrong Python after activation: $(command -v python)" >&2
  return 1
fi
echo "Activated $CONDA_ENV"
echo "Python: $(command -v python)"
