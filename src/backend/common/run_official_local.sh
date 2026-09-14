#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"

exec conda run --no-capture-output -n qwen_saliency \
  "$project_dir/.venv_official/bin/python" -m src.backend.common.run_official_experiment "$@"
