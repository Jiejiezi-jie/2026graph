import math

import networkx as nx

from app.services.graph_service import GraphService


def test_full_graph_has_all_nodes_edges_and_reuses_layout(tmp_path, monkeypatch):
    graph = nx.barabasi_albert_graph(310, 3, seed=42)
    graph = nx.relabel_nodes(graph, str)
    path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, path)
    service = GraphService(path)
    result = service.full()
    assert len(result.nodes) == 310
    assert len(result.edges) == graph.number_of_edges()
    assert not result.truncated
    assert len({n.id for n in result.nodes}) == len(result.nodes)
    assert all(math.isfinite(n.position["x"]) and math.isfinite(n.position["y"]) for n in result.nodes)
    positions = {n.id: n.position for n in result.nodes}
    from app.services import graph_layout
    monkeypatch.setattr(graph_layout, "compute_positions", lambda g: (_ for _ in ()).throw(AssertionError("layout rerun")))
    assert {n.id: n.position for n in GraphService(path).full().nodes} == positions


def test_query_hit_ids_are_complete_even_when_record_graph_is_bounded(tmp_path):
    graph = nx.complete_graph(90)
    graph = nx.relabel_nodes(graph, str)
    path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, path)
    service = GraphService(path)
    hits = [{"entity_name": str(i)} for i in range(90)]
    result = service.related(hits, [{"src_id": "88", "tgt_id": "89"}], max_nodes=5)
    assert len(result.nodes) == 5
    assert set(result.hit_node_ids) == set(graph)
    assert result.hit_edge_pairs == [["88", "89"]]
    assert all(not e.retrieved for e in result.edges)
    full = service.full()
    assert not any(n.retrieved for n in full.nodes)


def test_layout_invalidates_when_graph_changes(tmp_path):
    path = tmp_path / "graph.graphml"
    nx.write_graphml(nx.path_graph(["a", "b"]), path)
    service = GraphService(path)
    first = service.full()
    nx.write_graphml(nx.path_graph(["a", "b", "c"]), path)
    second = service.full()
    assert first.layout_key != second.layout_key
    assert len(second.nodes) == 3


def test_full_graph_keeps_isolated_and_directed_edges(tmp_path):
    graph = nx.DiGraph()
    graph.add_nodes_from(["isolated", "A", "B"])
    graph.add_edge("A", "B")
    graph.add_edge("B", "A")
    path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, path)
    result = GraphService(path).full()
    assert len(result.nodes) == 3 and len(result.edges) == 2
    assert all(e.directed for e in result.edges)


def test_layout_separates_dense_hub_neighbors():
    from app.services.graph_layout import compute_positions
    graph = nx.barabasi_albert_graph(250, 5, seed=42)
    positions = list(compute_positions(graph).values())
    closest = min(math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                  for i, a in enumerate(positions) for b in positions[i + 1:])
    assert closest > 50  # 38-unit circles retain a visible gap around dense hubs.


def test_isolated_nodes_form_even_outer_circle():
    from app.services.graph_layout import compute_positions
    graph = nx.path_graph(["a", "b", "c"])
    graph.add_edge("small-a", "small-b")
    isolated = [f"isolated-{i:02d}" for i in range(12)]
    graph.add_nodes_from(isolated)
    positions = compute_positions(graph)
    interior = [positions[n] for n in graph if graph.degree(n)]
    cx = (min(p["x"] for p in interior) + max(p["x"] for p in interior)) / 2
    cy = (min(p["y"] for p in interior) + max(p["y"] for p in interior)) / 2
    radius = [math.hypot(positions[n]["x"] - cx, positions[n]["y"] - cy) for n in isolated]
    assert max(radius) - min(radius) < .01
    assert min(radius) > max(math.hypot(p["x"] - cx, p["y"] - cy) for p in interior) + 100
    angles = sorted(math.atan2(positions[n]["y"] - cy, positions[n]["x"] - cx) for n in isolated)
    gaps = [(angles[(i+1) % len(angles)] - angles[i]) % math.tau for i in range(len(angles))]
    assert max(gaps) - min(gaps) < .001
    assert len(positions) == len(graph) and graph.number_of_edges() == 3


def test_all_isolated_graph_uses_circle_and_handles_empty_graph():
    from app.services.graph_layout import compute_positions
    assert compute_positions(nx.Graph()) == {}
    for size in (1, 2, 50):
        positions = compute_positions(nx.empty_graph(size))
        radii = [math.hypot(p["x"], p["y"]) for p in positions.values()]
        assert len(positions) == size
        assert max(radii) - min(radii) < .01
