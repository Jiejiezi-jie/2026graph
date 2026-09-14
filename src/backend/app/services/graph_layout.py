"""Cached display coordinates. Never changes the retrieval graph or vectors."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import networkx as nx


def compute_positions(graph):
    if not graph:
        return {}
    simple = nx.Graph(graph)
    isolated = sorted(nx.isolates(simple), key=str)
    connected = simple.subgraph(set(simple) - set(isolated))
    groups = list(nx.community.louvain_communities(connected, weight=None, seed=42)) if connected else []
    groups.sort(key=lambda group: (-len(group), min(map(str, group))))
    positions, placed = {}, []
    for group in groups:
        members = sorted(group, key=lambda n: (-simple.degree(n), str(n)))
        radius = max(100, math.sqrt(len(members)) * 72)
        # Pack communities without overlapping their bounding disks.
        step = 0
        while True:
            angle = step * 2.399963
            distance = 90 * math.sqrt(step)
            cx, cy = math.cos(angle) * distance, math.sin(angle) * distance
            if all(math.hypot(cx - x, cy - y) >= radius + rad + 100 for x, y, rad in placed):
                break
            step += 1
        placed.append((cx, cy, radius))
        local = simple.subgraph(members)
        if len(members) <= 500 and local.number_of_edges():
            coords = nx.spring_layout(local, seed=42, weight=None, iterations=45, scale=radius)
        else:
            coords = {n: (math.cos(i * 2.399963) * 100 * math.sqrt(i),
                          math.sin(i * 2.399963) * 100 * math.sqrt(i)) for i, n in enumerate(members)}
        for n, (x, y) in coords.items():
            positions[str(n)] = {"x": round(cx + float(x), 3), "y": round(cy + float(y), 3)}
    _separate_nodes(positions)
    if isolated:
        interior = list(positions.values())
        cx = (min(p["x"] for p in interior) + max(p["x"] for p in interior)) / 2 if interior else 0
        cy = (min(p["y"] for p in interior) + max(p["y"] for p in interior)) / 2 if interior else 0
        reach = max((math.hypot(p["x"] - cx, p["y"] - cy) for p in interior), default=0)
        # Only degree-zero nodes belong on this ring; small connected components
        # stay inside. The circle is a layout, not an invented relationship.
        radius = max(reach + 220, len(isolated) * 76 / math.tau, 180)
        for i, node in enumerate(isolated):
            angle = math.tau * i / len(isolated) - math.pi / 2
            positions[str(node)] = {"x": round(cx + radius * math.cos(angle), 3),
                                    "y": round(cy + radius * math.sin(angle), 3)}
    return positions


def _separate_nodes(positions, gap=66):
    """Resolve dense-hub overlaps once using a local spatial grid."""
    buckets = {}
    for point in positions.values():
        original_x, original_y = point["x"], point["y"]
        step = 0
        while True:
            angle, distance = step * 2.399963, 8 * math.sqrt(step)
            x, y = original_x + math.cos(angle) * distance, original_y + math.sin(angle) * distance
            cell = (math.floor(x / gap), math.floor(y / gap))
            nearby = (p for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      for p in buckets.get((cell[0] + dx, cell[1] + dy), []))
            if all((x - px) ** 2 + (y - py) ** 2 >= gap ** 2 for px, py in nearby):
                point.update(x=round(x, 3), y=round(y, 3))
                buckets.setdefault(cell, []).append((x, y))
                break
            step += 1


def cached_positions(path: Path, graph):
    key = hashlib.sha256(b"display-layout-v2-isolate-ring\0" + path.read_bytes()).hexdigest()
    cache = path.with_name(path.stem + ".display-layout.json")
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        coords = data["positions"]
        if data["key"] == key and set(coords) == set(map(str, graph)) and all(
            math.isfinite(p[axis]) for p in coords.values() for axis in ("x", "y")
        ):
            return key, coords
    except (OSError, ValueError, KeyError, TypeError):
        pass
    coords = compute_positions(graph)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump({"key": key, "positions": coords}, handle, ensure_ascii=False)
        os.replace(temporary, cache)
    except OSError:
        # Read-only bundles can still be displayed; GraphService keeps memory cache.
        pass
    finally:
        if temporary and os.path.isfile(temporary):
            os.unlink(temporary)
    return key, coords
