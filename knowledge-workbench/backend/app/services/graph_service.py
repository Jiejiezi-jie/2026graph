import hashlib
from pathlib import Path

import networkx as nx

from app.domain.errors import AppError
from app.retrieval.contracts import GraphData, GraphEdge, GraphNode


class GraphService:
    """Read-only NetworkXStorage view. Highlight actual hits, not guessed paths."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._stamp = None
        self._graph = None

    def _load(self):
        if not self.path.is_file():
            raise AppError("GRAPH_NOT_FOUND", "尚无可读取的 GraphML，请先建立索引。")
        stamp = (self.path.stat().st_mtime_ns, self.path.stat().st_size)
        if stamp != self._stamp:
            try:
                self._graph = nx.read_graphml(self.path)
                self._stamp = stamp
            except Exception as exc:
                raise AppError("INVALID_GRAPH", "GraphML 无法读取，请检查索引完整性。") from exc
        return self._graph

    def preview(self, max_nodes: int = 60) -> GraphData:
        graph = self._load()
        nodes = sorted(graph, key=lambda n: (-graph.degree(n), str(n)))[:max_nodes]
        return self._render(graph, nodes, set(), set(), len(graph) > len(nodes))

    def related(self, entities: list[dict], relationships: list[dict], max_nodes: int = 80) -> GraphData:
        graph = self._load()
        # Exact identifiers only: no fuzzy joining of unrelated entities.
        direct = {str(e.get("entity_name", "")) for e in entities}
        pairs = {(str(r.get("src_id", "")), str(r.get("tgt_id", ""))) for r in relationships}
        seeds = direct | {n for pair in pairs for n in pair}
        present = sorted(seeds & set(graph))
        selected = present[:max_nodes]
        candidates = set(present)
        for node in present:
            candidates.update(graph.neighbors(node))
        for node in sorted(candidates - set(selected)):
            if len(selected) >= max_nodes:
                break
            selected.append(node)
        result = self._render(graph, selected, seeds, pairs, len(candidates) > len(selected))
        missing = len(seeds - set(graph))
        if missing:
            result.warnings.append(f"{missing} 个检索实体未能与 GraphML 精确匹配。")
        return result

    @staticmethod
    def _render(graph, selected, direct, pairs, truncated):
        nodes = [
            GraphNode(id=str(n), label=str(n), type=str(graph.nodes[n].get("entity_type", "UNKNOWN")),
                      description=str(graph.nodes[n].get("description", "")), retrieved=n in direct)
            for n in selected
        ]
        edges = []
        all_edges = list(graph.subgraph(selected).edges(data=True))
        # Prioritize retrieved edges when applying the display cap.
        def hit(a, b):
            return (a, b) in pairs or (not graph.is_directed() and (b, a) in pairs)
        all_edges.sort(key=lambda e: (not hit(e[0], e[1]), str(e[0]), str(e[1])))
        for i, (a, b, attrs) in enumerate(all_edges[:200]):
            digest = hashlib.sha256(f"{a}\0{b}\0{i}".encode()).hexdigest()[:20]
            edges.append(GraphEdge(id="edge:" + digest, source=str(a), target=str(b),
                                   label=str(attrs.get("keywords", "")),
                                   description=str(attrs.get("description", "")),
                                   retrieved=hit(a, b), directed=graph.is_directed()))
        return GraphData(nodes=nodes, edges=edges, truncated=truncated or len(all_edges) > 200,
                         total_nodes=graph.number_of_nodes(), total_edges=graph.number_of_edges())
