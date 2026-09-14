#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
benchmark_dir="${1:-$project_dir/data/vendor/GraphRAG-Benchmark}"

cd "$project_dir"
python result/phase1_proxy/code/run_experiment.py \
  --benchmark-dir "$benchmark_dir" \
  --config configs/phase1.json \
  --output-dir result/phase1_proxy/outputs
