#!/bin/bash
# Source from the repository root:
#   source activate_env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this shortcut instead of executing it:" >&2
  echo "  source activate_env.sh" >&2
  exit 1
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/activate_paper_env.sh"
