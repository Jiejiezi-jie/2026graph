from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and export the completed official LightRAG GraphML"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("results_official/indexes/lightrag"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/lightrag")
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    graph_path = args.index_dir / "medical" / "graph_chunk_entity_relation.graphml"
    index_manifest_path = args.index_dir / "official_index_manifest.json"
    if not graph_path.exists() or not index_manifest_path.exists():
        raise FileNotFoundError("LightRAG graph or completed-index manifest is missing")
    if graph_path.stat().st_size == 0:
        raise RuntimeError("Refusing to export an empty GraphML file")

    graph = nx.read_graphml(graph_path)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise RuntimeError("Refusing to export a graph without nodes and edges")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported_graph = args.output_dir / "graph_chunk_entity_relation.graphml"
    shutil.copy2(graph_path, exported_graph)
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "status": "complete_official_lightrag_graph",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": "HKUDS/LightRAG@28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c",
        "source_index_manifest": index_manifest,
        "graph_file": exported_graph.name,
        "sha256": sha256(exported_graph),
        "bytes": exported_graph.stat().st_size,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
