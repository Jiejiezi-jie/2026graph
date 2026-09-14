"""Export every PathRAG node and edge to a self-contained offline explorer."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, default=Path('result/api/indexes/pathrag/graph_chunk_entity_relation.graphml'))
    parser.add_argument('--output', type=Path, default=Path('result/api/graph_preview/pathrag_full_explorer.html'))
    args = parser.parse_args()
    raw = args.graph.read_bytes()
    graph = nx.parse_graphml(raw.decode('utf-8'))
    print(f'Layout: {len(graph)} nodes, {graph.number_of_edges()} edges', flush=True)
    positions = nx.spring_layout(graph.to_undirected(), seed=42, iterations=60, weight=None, scale=1800)
    ids = {name: str(i) for i, name in enumerate(graph)}
    palette = ['#007f86', '#cf4770', '#5a68c9', '#37914c', '#b67b13', '#9670a0', '#357bac', '#a4513c']
    types = sorted({str(a.get('entity_type', 'UNKNOWN')).strip('"') for _, a in graph.nodes(data=True)})
    colors = {t: palette[i % len(palette)] for i, t in enumerate(types)}
    elements = []
    for name, attrs in graph.nodes(data=True):
        kind = str(attrs.get('entity_type', 'UNKNOWN')).strip('"')
        elements.append({'data': {'id': ids[name], 'label': name.strip('"'), 'original_id': name, 'kind': kind, 'color': colors[kind], 'degree': graph.degree(name), 'size': 5 + min(22, graph.degree(name) ** .5 * 1.8), 'attributes': attrs}, 'position': {'x': float(positions[name][0]), 'y': float(positions[name][1])}})
    for i, (source, target, attrs) in enumerate(graph.edges(data=True)):
        elements.append({'data': {'id': f'e{i}', 'source': ids[source], 'target': ids[target], 'attributes': attrs}})
    payload = {'elements': elements, 'types': colors, 'nodes': len(graph), 'edges': graph.number_of_edges(), 'directed': graph.is_directed(), 'source': args.graph.as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(), 'exported': datetime.now(timezone.utc).isoformat()}
    template = Path(__file__).with_name('pathrag_explorer.html').read_text(encoding='utf-8')
    library = args.output.parent.joinpath('cytoscape.min.js').read_text(encoding='utf-8')
    html = template.replace('/*__LIBRARY__*/', library).replace('/*__DATA__*/', json.dumps(payload, ensure_ascii=True).replace('<', '\\u003c'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k not in ('elements', 'types')}, indent=2))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
