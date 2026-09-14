#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"

python_bin="${OFFICIAL_PYTHON:-$project_dir/.venv_official/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="${PYTHON:-python}"
fi

exec "$python_bin" -m src.backend.common.run_official_experiment "$@"
