#!/usr/bin/env bash
set -euo pipefail

build_pid="${1:?usage: publish_lightrag_graph.sh BUILD_PID [TARGET_REPO]}"
target_repo="${2:-/home/user/wangyuhan/2026graph}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while kill -0 "$build_pid" 2>/dev/null; do
  sleep 30
done

graph="$project_dir/artifacts/lightrag/graph_chunk_entity_relation.graphml"
manifest="$project_dir/artifacts/lightrag/manifest.json"
if [[ ! -s "$graph" || ! -s "$manifest" ]]; then
  echo "Completed LightRAG graph artifacts are missing; refusing to publish" >&2
  exit 1
fi
if [[ ! -d "$target_repo/.git" ]]; then
  echo "Target is not a Git repository: $target_repo" >&2
  exit 1
fi

graph_bytes="$(stat -c '%s' "$graph")"
if (( graph_bytes >= 95000000 )); then
  echo "GraphML is too large for the configured direct Git workflow: $graph_bytes bytes" >&2
  exit 1
fi

mkdir -p "$target_repo/artifacts/lightrag" "$target_repo/scripts" "$target_repo/configs"
cp "$graph" "$target_repo/artifacts/lightrag/graph_chunk_entity_relation.graphml"
cp "$manifest" "$target_repo/artifacts/lightrag/manifest.json"
cp "$project_dir/.gitignore" "$target_repo/.gitignore"
cp "$project_dir/configs/official_local.json" "$target_repo/configs/official_local.json"
cp "$project_dir/scripts/run_official_experiment.py" "$target_repo/scripts/run_official_experiment.py"
cp "$project_dir/scripts/export_lightrag_graph.py" "$target_repo/scripts/export_lightrag_graph.py"

git -C "$target_repo" add \
  artifacts/lightrag/graph_chunk_entity_relation.graphml \
  artifacts/lightrag/manifest.json \
  .gitignore \
  configs/official_local.json \
  scripts/run_official_experiment.py \
  scripts/export_lightrag_graph.py
git -C "$target_repo" diff --cached --check
git -C "$target_repo" commit -m "Add completed LightRAG knowledge graph"
git -C "$target_repo" push origin main
