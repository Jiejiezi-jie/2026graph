#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
target_dir="$project_dir/data/vendor/GraphRAG-Benchmark"

if [[ -d "$target_dir/.git" ]]; then
  echo "GraphRAG-Bench already exists at $target_dir"
  exit 0
fi

mkdir -p "$(dirname "$target_dir")"
git clone --depth 1 https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git "$target_dir"
echo "Downloaded GraphRAG-Bench to $target_dir"

