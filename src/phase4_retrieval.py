"""Shared-data retrievers for the course project's phase four.

These are deliberately small, auditable course implementations inspired by
LightRAG local retrieval and PathRAG path retrieval.  They are not complete
reproductions of either upstream project and never generate an answer.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import networkx as nx
import numpy as np


ALLOWED_STATUSES = {
    "success",
    "empty",
    "insufficient_entities",
    "disconnected",
    "no_path",
}
RETRIEVER_NAMES = ("vector", "neighborhood", "path")
FORBIDDEN_RESULT_KEYS = {
    "answer",
    "evidence",
    "evidence_triple",
    "label",
    "question_type",
}
REQUIRED_RESULT_KEYS = {
    "query_id",
    "query",
    "retriever",
    "status",
    "linked_entities",
    "retrieved_chunks",
    "retrieved_nodes",
    "retrieved_edges",
    "paths",
    "diagnostics",
}


class Phase4Error(RuntimeError):
    """Raised when phase-four inputs or invariants are invalid."""


class QueryEncoder(Protocol):
    dimension: int

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Return normalized float32 query vectors."""


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def jsonl_dump(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def structural_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(strip_timing_fields(value)).encode("utf-8"))


def strip_timing_fields(value: Any) -> Any:
    """Remove runtime-only fields before deterministic comparisons."""
    if isinstance(value, dict):
        return {
            key: strip_timing_fields(item)
            for key, item in value.items()
            if not (
                key.endswith("_ms")
                or key in {"latency", "latencies", "timing", "timings"}
            )
        }
    if isinstance(value, list):
        return [strip_timing_fields(item) for item in value]
    return value


def normalize_phrase(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"[^\w']+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def phrase_in_query(query: str, phrase: str) -> bool:
    query_normalized = f" {normalize_phrase(query)} "
    phrase_normalized = normalize_phrase(phrase)
    return bool(phrase_normalized) and f" {phrase_normalized} " in query_normalized


def rounded_score(value: float) -> float:
    return round(float(value), 8)


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase4Error(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase4Error(f"invalid JSON: {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise Phase4Error(f"missing file: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase4Error(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise Phase4Error(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def load_retrieval_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    required_fixed = {
        "random_seed": 42,
        "top_k_chunks": 5,
        "max_seed_entities": 4,
        "entity_similarity_threshold": 0.55,
        "neighborhood_hops": 1,
        "max_path_hops": 3,
        "max_paths_per_pair": 3,
    }
    mismatches = {
        key: config.get(key)
        for key, expected in required_fixed.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise Phase4Error(f"phase-four fixed configuration mismatch: {mismatches}")
    if config.get("excluded_path_relations") != ["AUTHORED_BY", "PORTRAYED_BY"]:
        raise Phase4Error("excluded_path_relations must be AUTHORED_BY and PORTRAYED_BY")
    embedding = config.get("embedding", {})
    if embedding.get("model_name") != "BAAI/bge-small-en-v1.5":
        raise Phase4Error("default embedding model must be BAAI/bge-small-en-v1.5")
    if not embedding.get("query_prefix") or embedding.get("pooling") != "cls":
        raise Phase4Error("BGE query prefix and CLS pooling must be configured")
    for group_name in ("neighborhood_scoring", "path_scoring"):
        weights = [
            float(value)
            for key, value in config[group_name].items()
            if key.endswith("weight")
        ]
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise Phase4Error(f"{group_name} weights must sum to one")
    return config


def resolve_config_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    inputs = {key: (root / value).resolve() for key, value in config["inputs"].items()}
    expected_suffixes = {
        "chunks": "data/processed/phase2/chunks.jsonl",
        "nodes": "data/processed/phase3/nodes.jsonl",
        "edges": "data/processed/phase3/edges.jsonl",
        "chunk_entity_map": "data/processed/phase3/chunk_entity_map.json",
        "train_questions": "data/processed/phase1/train_questions.jsonl",
    }
    for key, suffix in expected_suffixes.items():
        if not inputs[key].as_posix().endswith(suffix):
            raise Phase4Error(f"invalid phase-four input path for {key}: {inputs[key]}")
    indices_dir = (root / config["indices"]["directory"]).resolve()
    outputs_dir = (root / config["outputs"]["directory"]).resolve()
    return {
        **inputs,
        "indices_dir": indices_dir,
        "chunk_embeddings": indices_dir / config["indices"]["chunk_embeddings"],
        "entity_embeddings": indices_dir / config["indices"]["entity_embeddings"],
        "index_metadata": indices_dir / config["indices"]["metadata"],
        "outputs_dir": outputs_dir,
        "results": outputs_dir / config["outputs"]["results"],
        "diagnostics": outputs_dir / config["outputs"]["diagnostics"],
        "manifest": outputs_dir / config["outputs"]["manifest"],
        "report": (root / config["outputs"]["report"]).resolve(),
    }


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    projected = []
    for row in rows:
        required = {"chunk_id", "text", "section_id"}
        if not required <= set(row):
            raise Phase4Error(f"chunk missing fields: {row.get('chunk_id')}")
        projected.append({key: row[key] for key in ("chunk_id", "text", "section_id")})
    if len(projected) != 62 or len({row["chunk_id"] for row in projected}) != 62:
        raise Phase4Error("phase four expects exactly 62 unique chunks")
    return projected


def load_train_questions(path: Path) -> list[dict[str, str]]:
    """Project only non-supervisory fields from the allowed training split."""
    rows = read_jsonl(path)
    projected: list[dict[str, str]] = []
    for row in rows:
        required = {"id", "question", "question_type"}
        if not required <= set(row):
            raise Phase4Error("training question row is missing id/question/question_type")
        projected.append({key: str(row[key]) for key in ("id", "question", "question_type")})
    if len(projected) != 62 or len({row["id"] for row in projected}) != 62:
        raise Phase4Error("phase four expects exactly 62 unique training questions")
    allowed_types = {"Fact Retrieval", "Complex Reasoning"}
    if {row["question_type"] for row in projected} - allowed_types:
        raise Phase4Error("unexpected question type in training questions")
    return projected


def load_graph_inputs(
    paths: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    chunks = load_chunks(paths["chunks"])
    nodes = read_jsonl(paths["nodes"])
    edges = read_jsonl(paths["edges"])
    chunk_map = read_json(paths["chunk_entity_map"])
    if len(nodes) != 124 or len(edges) != 80:
        raise Phase4Error("phase-three graph size does not match the accepted manifest")
    node_ids = [row.get("node_id") for row in nodes]
    edge_ids = [row.get("edge_id") for row in edges]
    if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)):
        raise Phase4Error("phase-three node or edge IDs are not unique")
    valid_nodes = set(node_ids)
    valid_chunks = {row["chunk_id"] for row in chunks}
    if any(
        edge.get("source_id") not in valid_nodes
        or edge.get("target_id") not in valid_nodes
        or not set(edge.get("source_chunk_ids", [])) <= valid_chunks
        for edge in edges
    ):
        raise Phase4Error("phase-three edge has invalid endpoint or chunk source")
    mapped_chunks = chunk_map.get("chunks", {})
    if set(mapped_chunks) != valid_chunks:
        raise Phase4Error("chunk_entity_map does not account for all phase-two chunks")
    return chunks, nodes, edges, chunk_map


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


class BGEEncoder:
    """Minimal BGE encoder using Transformers and documented CLS pooling."""

    def __init__(self, config: Mapping[str, Any], seed: int):
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise Phase4Error(
                "PyTorch and Transformers are required for BAAI/bge-small-en-v1.5"
            ) from exc
        seed_everything(seed)
        self._torch = torch
        self.transformers_version = transformers.__version__
        self.torch_version = torch.__version__
        self.config = dict(config)
        self.model_path = Path(self.config["local_model_path"]).resolve()
        if not self.model_path.is_dir():
            raise Phase4Error(
                f"embedding model download missing: {self.model_path}; refusing substitution"
            )
        weights_path = self.model_path / self.config["weights_file"]
        if not weights_path.is_file():
            raise Phase4Error(
                f"embedding weights missing: {weights_path}; refusing substitution"
            )
        actual_weight_hash = sha256_file(weights_path)
        if actual_weight_hash != self.config["weights_sha256"]:
            raise Phase4Error("embedding weight checksum does not match retrieval_config.json")
        actual_revision = git_revision(self.model_path)
        if actual_revision and actual_revision != self.config["model_revision"]:
            raise Phase4Error("embedding repository revision does not match configuration")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self.device = torch.device(self.config["device"])
        self.model.to(self.device)
        self.model.eval()
        self.dimension = int(self.model.config.hidden_size)
        self.actual_revision = actual_revision or self.config["model_revision"]
        self.weights_sha256 = actual_weight_hash

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        batch_size = int(self.config["batch_size"])
        torch = self._torch
        for start in range(0, len(texts), batch_size):
            prepared = [prefix + text for text in texts[start : start + batch_size]]
            encoded = self.tokenizer(
                prepared,
                padding=True,
                truncation=True,
                max_length=int(self.config["max_length"]),
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                vectors = hidden[:, 0]
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            batches.append(vectors.detach().cpu().to(torch.float32).numpy())
        result = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        return result

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.config["query_prefix"])

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.config.get("document_prefix", ""))

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.config["model_name"],
            "model_revision": self.actual_revision,
            "weights_file": self.config["weights_file"],
            "weights_sha256": self.weights_sha256,
            "vector_dimension": self.dimension,
            "pooling": self.config["pooling"],
            "normalize": bool(self.config["normalize"]),
            "query_prefix": self.config["query_prefix"],
            "document_prefix": self.config.get("document_prefix", ""),
            "max_length": int(self.config["max_length"]),
            "device": self.config["device"],
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "numpy_version": np.__version__,
            "networkx_version": nx.__version__,
        }


def entity_document(node: Mapping[str, Any]) -> str:
    aliases = "; ".join(str(value) for value in node.get("aliases", []))
    return (
        f"Name: {node['name']}. Type: {node['type']}. "
        f"Aliases: {aliases}. Description: {node.get('description', '')}"
    )


def validate_normalized_embeddings(array: np.ndarray, expected_rows: int) -> None:
    if array.ndim != 2 or len(array) != expected_rows or array.dtype != np.float32:
        raise Phase4Error("embedding matrix shape or dtype is invalid")
    norms = np.linalg.norm(array, axis=1)
    if not np.all(np.isfinite(array)) or not np.allclose(norms, 1.0, atol=2e-5):
        raise Phase4Error("embedding vectors must be finite and L2 normalized")


def build_indices(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_retrieval_config(config_path)
    paths = resolve_config_paths(root, config)
    chunks, nodes, _edges, _chunk_map = load_graph_inputs(paths)
    before_hashes = {
        key: sha256_file(paths[key])
        for key in ("chunks", "nodes", "edges", "chunk_entity_map")
    }
    encoder = BGEEncoder(config["embedding"], config["random_seed"])
    chunk_embeddings = encoder.encode_documents([row["text"] for row in chunks])
    entity_embeddings = encoder.encode_documents([entity_document(row) for row in nodes])
    validate_normalized_embeddings(chunk_embeddings, len(chunks))
    validate_normalized_embeddings(entity_embeddings, len(nodes))
    write_npy_atomic(paths["chunk_embeddings"], chunk_embeddings)
    write_npy_atomic(paths["entity_embeddings"], entity_embeddings)
    metadata = {
        "schema_version": config["schema_version"],
        "random_seed": config["random_seed"],
        "embedding": encoder.metadata(),
        "input_hashes": before_hashes,
        "chunk_count": len(chunks),
        "entity_count": len(nodes),
        "chunk_ids": [row["chunk_id"] for row in chunks],
        "entity_ids": [row["node_id"] for row in nodes],
        "index_files": {
            "chunk_embeddings": {
                "path": paths["chunk_embeddings"].relative_to(root).as_posix(),
                "sha256": sha256_file(paths["chunk_embeddings"]),
            },
            "entity_embeddings": {
                "path": paths["entity_embeddings"].relative_to(root).as_posix(),
                "sha256": sha256_file(paths["entity_embeddings"]),
            },
        },
    }
    write_text_atomic(paths["index_metadata"], canonical_json(metadata, pretty=True))
    after_hashes = {
        key: sha256_file(paths[key])
        for key in ("chunks", "nodes", "edges", "chunk_entity_map")
    }
    if before_hashes != after_hashes:
        raise Phase4Error("a phase-two or phase-three input changed during indexing")
    return metadata


def load_index_arrays(
    root: Path,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    chunks: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = read_json(paths["index_metadata"])
    for key in ("chunks", "nodes", "edges", "chunk_entity_map"):
        if metadata["input_hashes"].get(key) != sha256_file(paths[key]):
            raise Phase4Error(f"retrieval index is stale for {key}")
    if metadata["embedding"]["model_name"] != config["embedding"]["model_name"]:
        raise Phase4Error("retrieval index uses a different embedding model")
    chunk_vectors = np.load(paths["chunk_embeddings"], allow_pickle=False)
    entity_vectors = np.load(paths["entity_embeddings"], allow_pickle=False)
    validate_normalized_embeddings(chunk_vectors, len(chunks))
    validate_normalized_embeddings(entity_vectors, len(nodes))
    if metadata["chunk_ids"] != [row["chunk_id"] for row in chunks]:
        raise Phase4Error("chunk embedding order mismatch")
    if metadata["entity_ids"] != [row["node_id"] for row in nodes]:
        raise Phase4Error("entity embedding order mismatch")
    return chunk_vectors, entity_vectors, metadata


class EntityLinker:
    def __init__(self, resources: "RetrievalResources"):
        self.resources = resources

    def link(self, query: str, query_vector: np.ndarray) -> list[dict[str, Any]]:
        similarities = self.resources.entity_embeddings @ query_vector
        candidates: list[dict[str, Any]] = []
        threshold = float(self.resources.config["entity_similarity_threshold"])
        for index, node in enumerate(self.resources.nodes):
            lexical_score = 0.0
            method = "semantic"
            matched_term: str | None = None
            if phrase_in_query(query, node["name"]):
                lexical_score = 1.0
                method = "canonical"
                matched_term = node["name"]
            else:
                matching_aliases = sorted(
                    {
                        str(alias)
                        for alias in node.get("aliases", [])
                        if phrase_in_query(query, str(alias))
                    },
                    key=lambda value: (-len(normalize_phrase(value)), value.casefold(), value),
                )
                if matching_aliases:
                    lexical_score = 0.98
                    method = "alias"
                    matched_term = matching_aliases[0]
            semantic_score = float(similarities[index])
            if lexical_score == 0.0 and semantic_score < threshold:
                continue
            score = max(lexical_score, semantic_score)
            candidates.append(
                {
                    "node_id": node["node_id"],
                    "name": node["name"],
                    "type": node["type"],
                    "score": rounded_score(score),
                    "semantic_score": rounded_score(semantic_score),
                    "method": method,
                    "matched_term": matched_term,
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["score"],
                0 if row["method"] == "canonical" else 1 if row["method"] == "alias" else 2,
                -row["semantic_score"],
                row["node_id"],
            )
        )
        return candidates[: int(self.resources.config["max_seed_entities"])]


class RetrievalResources:
    def __init__(
        self,
        *,
        chunks: Sequence[Mapping[str, Any]],
        nodes: Sequence[Mapping[str, Any]],
        edges: Sequence[Mapping[str, Any]],
        chunk_entity_map: Mapping[str, Any],
        chunk_embeddings: np.ndarray,
        entity_embeddings: np.ndarray,
        encoder: QueryEncoder,
        config: Mapping[str, Any],
    ):
        self.chunks = [dict(row) for row in chunks]
        self.nodes = [dict(row) for row in nodes]
        self.edges = [dict(row) for row in edges]
        self.chunk_entity_map = dict(chunk_entity_map)
        self.chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)
        self.entity_embeddings = np.asarray(entity_embeddings, dtype=np.float32)
        self.encoder = encoder
        self.config = dict(config)
        validate_normalized_embeddings(self.chunk_embeddings, len(self.chunks))
        validate_normalized_embeddings(self.entity_embeddings, len(self.nodes))
        if self.chunk_embeddings.shape[1] != self.entity_embeddings.shape[1]:
            raise Phase4Error("chunk and entity vector dimensions differ")
        self.chunk_by_id = {row["chunk_id"]: row for row in self.chunks}
        self.chunk_index = {row["chunk_id"]: index for index, row in enumerate(self.chunks)}
        self.node_by_id = {row["node_id"]: row for row in self.nodes}
        self.node_index = {row["node_id"]: index for index, row in enumerate(self.nodes)}
        self.edge_by_id = {row["edge_id"]: row for row in self.edges}
        self.map_node_chunks: dict[str, set[str]] = collections.defaultdict(set)
        for chunk_id, item in self.chunk_entity_map["chunks"].items():
            for reference in item.get("node_refs", []):
                self.map_node_chunks[reference["node_id"]].add(chunk_id)
        self.graph = nx.MultiGraph()
        self.path_graph = nx.MultiGraph()
        for node in self.nodes:
            self.graph.add_node(node["node_id"])
            self.path_graph.add_node(node["node_id"])
        excluded = set(self.config["excluded_path_relations"])
        for edge in self.edges:
            attributes = {
                "edge_id": edge["edge_id"],
                "relation": edge["relation"],
            }
            self.graph.add_edge(
                edge["source_id"], edge["target_id"], key=edge["edge_id"], **attributes
            )
            if edge["relation"] not in excluded:
                self.path_graph.add_edge(
                    edge["source_id"],
                    edge["target_id"],
                    key=edge["edge_id"],
                    **attributes,
                )
        self.linker = EntityLinker(self)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self.encoder.encode_queries(queries), dtype=np.float32)
        validate_normalized_embeddings(vectors, len(queries))
        if vectors.shape[1] != self.chunk_embeddings.shape[1]:
            raise Phase4Error("query and index vector dimensions differ")
        return vectors

    def chunk_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        return self.chunk_embeddings @ query_vector

    def chunks_for_node(self, node_id: str) -> set[str]:
        node = self.node_by_id[node_id]
        return set(node.get("source_chunk_ids", [])) | self.map_node_chunks.get(
            node_id, set()
        )

    def compact_node(self, node_id: str, **extra: Any) -> dict[str, Any]:
        node = self.node_by_id[node_id]
        return {
            "node_id": node_id,
            "name": node["name"],
            "type": node["type"],
            **extra,
        }

    def compact_edge(self, edge_id: str) -> dict[str, Any]:
        edge = self.edge_by_id[edge_id]
        return {
            "edge_id": edge_id,
            "source_id": edge["source_id"],
            "target_id": edge["target_id"],
            "source": edge["source"],
            "target": edge["target"],
            "relation": edge["relation"],
            "directed": bool(edge["directed"]),
        }


def load_resources(
    root: Path, config_path: Path
) -> tuple[RetrievalResources, dict[str, Any], dict[str, Path], dict[str, Any]]:
    config = load_retrieval_config(config_path)
    paths = resolve_config_paths(root, config)
    chunks, nodes, edges, chunk_map = load_graph_inputs(paths)
    chunk_vectors, entity_vectors, metadata = load_index_arrays(
        root, config, paths, chunks, nodes
    )
    encoder = BGEEncoder(config["embedding"], config["random_seed"])
    if encoder.dimension != metadata["embedding"]["vector_dimension"]:
        raise Phase4Error("loaded BGE dimension differs from index metadata")
    resources = RetrievalResources(
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        chunk_entity_map=chunk_map,
        chunk_embeddings=chunk_vectors,
        entity_embeddings=entity_vectors,
        encoder=encoder,
        config=config,
    )
    return resources, config, paths, metadata


def base_result(query_id: str, query: str, retriever: str) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "query": query,
        "retriever": retriever,
        "status": "empty",
        "linked_entities": [],
        "retrieved_chunks": [],
        "retrieved_nodes": [],
        "retrieved_edges": [],
        "paths": [],
        "diagnostics": {},
    }


def ranked_chunks(
    scores: Mapping[str, float], top_k: int
) -> list[dict[str, Any]]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [
        {"chunk_id": chunk_id, "rank": rank, "score": rounded_score(score)}
        for rank, (chunk_id, score) in enumerate(ordered, 1)
    ]


def validate_result(result: Mapping[str, Any], top_k: int) -> None:
    if set(result) != REQUIRED_RESULT_KEYS:
        raise Phase4Error(f"retrieval result fields are invalid: {sorted(result)}")
    if FORBIDDEN_RESULT_KEYS & set(result):
        raise Phase4Error("retrieval result contains supervision fields")
    if result["retriever"] not in RETRIEVER_NAMES:
        raise Phase4Error("unknown retriever name")
    if result["status"] not in ALLOWED_STATUSES:
        raise Phase4Error("unknown retrieval status")
    chunks = result["retrieved_chunks"]
    if len(chunks) > top_k:
        raise Phase4Error("retrieval result exceeds common chunk budget")
    if [row["rank"] for row in chunks] != list(range(1, len(chunks) + 1)):
        raise Phase4Error("retrieved chunk ranks are not consecutive")
    if result["retriever"] == "vector" and any(
        result[key] for key in ("linked_entities", "retrieved_nodes", "retrieved_edges", "paths")
    ):
        raise Phase4Error("VectorRetriever must not expose graph information")
    if result["retriever"] == "path" and result["status"] in {
        "insufficient_entities",
        "disconnected",
        "no_path",
    } and chunks:
        raise Phase4Error("PathRetriever failure must not fall back to vector chunks")


class VectorRetriever:
    def __init__(self, resources: RetrievalResources):
        self.resources = resources

    def retrieve(
        self, query_id: str, query: str, query_vector: np.ndarray | None = None
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if query_vector is None:
            query_vector = self.resources.encode_queries([query])[0]
        similarities = self.resources.chunk_similarities(query_vector)
        score_map = {
            chunk["chunk_id"]: float(similarities[index])
            for index, chunk in enumerate(self.resources.chunks)
        }
        result = base_result(query_id, query, "vector")
        result["retrieved_chunks"] = ranked_chunks(
            score_map, int(self.resources.config["top_k_chunks"])
        )
        result["status"] = "success" if result["retrieved_chunks"] else "empty"
        result["diagnostics"] = {
            "index_size": len(self.resources.chunks),
            "candidate_chunk_count": len(score_map),
            "latency_ms": round((time.perf_counter() - started) * 1000, 6),
        }
        validate_result(result, int(self.resources.config["top_k_chunks"]))
        return result


class NeighborhoodRetriever:
    def __init__(self, resources: RetrievalResources):
        self.resources = resources

    def retrieve(
        self,
        query_id: str,
        query: str,
        query_vector: np.ndarray | None = None,
        linked_entities: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if query_vector is None:
            query_vector = self.resources.encode_queries([query])[0]
        if linked_entities is None:
            linked_entities = self.resources.linker.link(query, query_vector)
        links = [dict(row) for row in linked_entities]
        result = base_result(query_id, query, "neighborhood")
        result["linked_entities"] = links
        similarities = self.resources.chunk_similarities(query_vector)
        node_scores: dict[str, float] = {}
        node_distances: dict[str, int] = {}
        hop_limit = int(self.resources.config["neighborhood_hops"])
        decay = float(self.resources.config["neighborhood_scoring"]["hop_decay"])
        seed_ids = {row["node_id"] for row in links}
        for link in links:
            seed_id = link["node_id"]
            lengths = nx.single_source_shortest_path_length(
                self.resources.graph, seed_id, cutoff=hop_limit
            )
            for node_id, distance in lengths.items():
                score = float(link["score"]) * (decay**distance)
                if score > node_scores.get(node_id, -math.inf):
                    node_scores[node_id] = score
                    node_distances[node_id] = distance
        included_edges: set[str] = set()
        for source_id in sorted(node_scores):
            for target_id, keyed in self.resources.graph[source_id].items():
                if target_id not in node_scores:
                    continue
                for edge_id in keyed:
                    included_edges.add(edge_id)
        candidate_strength: dict[str, float] = {}
        for node_id, score in node_scores.items():
            for chunk_id in self.resources.chunks_for_node(node_id):
                candidate_strength[chunk_id] = max(
                    candidate_strength.get(chunk_id, 0.0), score
                )
        for edge_id in included_edges:
            edge = self.resources.edge_by_id[edge_id]
            strength = max(
                node_scores.get(edge["source_id"], 0.0),
                node_scores.get(edge["target_id"], 0.0),
            )
            for chunk_id in edge.get("source_chunk_ids", []):
                candidate_strength[chunk_id] = max(
                    candidate_strength.get(chunk_id, 0.0), strength
                )
        weights = self.resources.config["neighborhood_scoring"]
        combined_scores = {
            chunk_id: float(weights["entity_weight"]) * strength
            + float(weights["chunk_similarity_weight"])
            * float(similarities[self.resources.chunk_index[chunk_id]])
            for chunk_id, strength in candidate_strength.items()
        }
        result["retrieved_chunks"] = ranked_chunks(
            combined_scores, int(self.resources.config["top_k_chunks"])
        )
        result["retrieved_nodes"] = [
            self.resources.compact_node(
                node_id,
                distance=node_distances[node_id],
                score=rounded_score(node_scores[node_id]),
                seed=node_id in seed_ids,
            )
            for node_id in sorted(
                node_scores,
                key=lambda item: (node_distances[item], -node_scores[item], item),
            )
        ]
        result["retrieved_edges"] = [
            self.resources.compact_edge(edge_id) for edge_id in sorted(included_edges)
        ]
        result["status"] = "success" if result["retrieved_chunks"] else "empty"
        result["diagnostics"] = {
            "seed_entity_count": len(links),
            "expanded_node_count": len(node_scores),
            "expanded_edge_count": len(included_edges),
            "candidate_chunk_count": len(candidate_strength),
            "vector_fallback_used": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 6),
        }
        validate_result(result, int(self.resources.config["top_k_chunks"]))
        return result


def path_edge_ids(
    graph: nx.MultiGraph, node_path: Sequence[str]
) -> tuple[list[str], list[str]]:
    edge_ids: list[str] = []
    relations: list[str] = []
    for source, target in itertools.pairwise(node_path):
        candidates = sorted(
            (
                (attributes["relation"], attributes["edge_id"])
                for attributes in graph[source][target].values()
            ),
            key=lambda item: (item[0], item[1]),
        )
        relation, edge_id = candidates[0]
        edge_ids.append(edge_id)
        relations.append(relation)
    return edge_ids, relations


class PathRetriever:
    def __init__(self, resources: RetrievalResources):
        self.resources = resources

    def retrieve(
        self,
        query_id: str,
        query: str,
        query_vector: np.ndarray | None = None,
        linked_entities: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if query_vector is None:
            query_vector = self.resources.encode_queries([query])[0]
        if linked_entities is None:
            linked_entities = self.resources.linker.link(query, query_vector)
        links = [dict(row) for row in linked_entities]
        result = base_result(query_id, query, "path")
        result["linked_entities"] = links
        base_diagnostics = {
            "seed_entity_count": len(links),
            "pair_count": math.comb(len(links), 2) if len(links) >= 2 else 0,
            "connected_pair_count": 0,
            "paths_considered": 0,
            "excluded_relations": list(self.resources.config["excluded_path_relations"]),
            "vector_fallback_used": False,
        }
        if len(links) < 2:
            result["status"] = "insufficient_entities"
            base_diagnostics["latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 6
            )
            result["diagnostics"] = base_diagnostics
            validate_result(result, int(self.resources.config["top_k_chunks"]))
            return result

        link_by_id = {row["node_id"]: row for row in links}
        paths: list[dict[str, Any]] = []
        cutoff = int(self.resources.config["max_path_hops"])
        per_pair = int(self.resources.config["max_paths_per_pair"])
        similarities = self.resources.chunk_similarities(query_vector)
        for left, right in itertools.combinations(links, 2):
            source_id, target_id = left["node_id"], right["node_id"]
            if not nx.has_path(self.resources.path_graph, source_id, target_id):
                continue
            base_diagnostics["connected_pair_count"] += 1
            # MultiGraph yields the same node sequence once per parallel-edge
            # combination.  The logical path is the node sequence; edge choice is
            # resolved deterministically by ``path_edge_ids`` below.
            node_paths = sorted(
                {
                    tuple(node_path)
                    for node_path in nx.all_simple_paths(
                        self.resources.path_graph,
                        source_id,
                        target_id,
                        cutoff=cutoff,
                    )
                },
                key=lambda row: (len(row), row),
            )
            for node_path in node_paths[:per_pair]:
                edge_ids, relations = path_edge_ids(
                    self.resources.path_graph, node_path
                )
                evidence_chunks: set[str] = set()
                for node_id in node_path:
                    evidence_chunks.update(self.resources.chunks_for_node(node_id))
                for edge_id in edge_ids:
                    evidence_chunks.update(
                        self.resources.edge_by_id[edge_id].get("source_chunk_ids", [])
                    )
                endpoint_score = (
                    float(left["score"]) + float(right["score"])
                ) / 2.0
                hops = len(node_path) - 1
                best_chunk_similarity = max(
                    (
                        float(similarities[self.resources.chunk_index[chunk_id]])
                        for chunk_id in evidence_chunks
                    ),
                    default=0.0,
                )
                weights = self.resources.config["path_scoring"]
                score = (
                    float(weights["endpoint_weight"]) * endpoint_score
                    + float(weights["length_weight"]) * (1.0 / hops)
                    + float(weights["chunk_similarity_weight"])
                    * best_chunk_similarity
                )
                paths.append(
                    {
                        "path_id": stable_id(
                            "p", *node_path, *edge_ids
                        ),
                        "source_id": source_id,
                        "target_id": target_id,
                        "node_ids": list(node_path),
                        "edge_ids": edge_ids,
                        "relations": relations,
                        "hops": hops,
                        "score": rounded_score(score),
                        "endpoint_score": rounded_score(endpoint_score),
                        "evidence_chunk_ids": sorted(evidence_chunks),
                    }
                )
        if base_diagnostics["connected_pair_count"] == 0:
            result["status"] = "disconnected"
            base_diagnostics["latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 6
            )
            result["diagnostics"] = base_diagnostics
            validate_result(result, int(self.resources.config["top_k_chunks"]))
            return result
        if not paths:
            result["status"] = "no_path"
            base_diagnostics["latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 6
            )
            result["diagnostics"] = base_diagnostics
            validate_result(result, int(self.resources.config["top_k_chunks"]))
            return result

        paths.sort(key=lambda row: (-row["score"], row["hops"], row["path_id"]))
        base_diagnostics["paths_considered"] = len(paths)
        selected_node_ids = {node_id for row in paths for node_id in row["node_ids"]}
        selected_edge_ids = {edge_id for row in paths for edge_id in row["edge_ids"]}
        candidate_scores: dict[str, float] = {}
        weights = self.resources.config["path_scoring"]
        for path in paths:
            path_length_score = 1.0 / int(path["hops"])
            for chunk_id in path["evidence_chunk_ids"]:
                score = (
                    float(weights["endpoint_weight"])
                    * float(path["endpoint_score"])
                    + float(weights["length_weight"]) * path_length_score
                    + float(weights["chunk_similarity_weight"])
                    * float(similarities[self.resources.chunk_index[chunk_id]])
                )
                candidate_scores[chunk_id] = max(
                    candidate_scores.get(chunk_id, -math.inf), score
                )
        result["paths"] = paths
        result["retrieved_chunks"] = ranked_chunks(
            candidate_scores, int(self.resources.config["top_k_chunks"])
        )
        result["retrieved_nodes"] = [
            self.resources.compact_node(
                node_id,
                linked=node_id in link_by_id,
                link_score=(
                    rounded_score(link_by_id[node_id]["score"])
                    if node_id in link_by_id
                    else None
                ),
            )
            for node_id in sorted(selected_node_ids)
        ]
        result["retrieved_edges"] = [
            self.resources.compact_edge(edge_id) for edge_id in sorted(selected_edge_ids)
        ]
        result["status"] = "success" if result["retrieved_chunks"] else "empty"
        base_diagnostics.update(
            {
                "candidate_chunk_count": len(candidate_scores),
                "selected_path_count": len(paths),
                "latency_ms": round((time.perf_counter() - started) * 1000, 6),
            }
        )
        result["diagnostics"] = base_diagnostics
        validate_result(result, int(self.resources.config["top_k_chunks"]))
        return result


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percent))


def summarize_group(
    question_ids: set[str], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected = [row for row in results if row["query_id"] in question_ids]
    by_retriever = {
        name: [row for row in selected if row["retriever"] == name]
        for name in RETRIEVER_NAMES
    }
    path_rows = by_retriever["path"]
    entity_link_success = sum(bool(row["linked_entities"]) for row in path_rows)
    at_least_two = sum(len(row["linked_entities"]) >= 2 for row in path_rows)
    same_component = sum(
        row["diagnostics"].get("connected_pair_count", 0) > 0 for row in path_rows
    )
    denominator = len(question_ids)
    retriever_stats: dict[str, Any] = {}
    for name, rows in by_retriever.items():
        latencies = [float(row["diagnostics"]["latency_ms"]) for row in rows]
        status_counts = dict(
            sorted(collections.Counter(row["status"] for row in rows).items())
        )
        retriever_stats[name] = {
            "status_counts": status_counts,
            "nonempty_rate": rounded_score(
                sum(bool(row["retrieved_chunks"]) for row in rows) / denominator
            ),
            "success_rate": rounded_score(
                sum(row["status"] == "success" for row in rows) / denominator
            ),
            "average_returned_chunks": rounded_score(
                statistics.fmean(len(row["retrieved_chunks"]) for row in rows)
            ),
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 6),
                "p50": round(percentile(latencies, 50), 6),
                "p95": round(percentile(latencies, 95), 6),
            },
        }
    return {
        "query_count": denominator,
        "entity_link_success_rate": rounded_score(entity_link_success / denominator),
        "linked_at_least_two_rate": rounded_score(at_least_two / denominator),
        "same_component_rate": rounded_score(same_component / denominator),
        "same_component_rate_among_multi_entity": rounded_score(
            same_component / at_least_two if at_least_two else 0.0
        ),
        "neighborhood_nonempty_rate": retriever_stats["neighborhood"]["nonempty_rate"],
        "path_found_rate": rounded_score(
            sum(bool(row["paths"]) for row in path_rows) / denominator
        ),
        "path_failure_reasons": {
            key: value
            for key, value in retriever_stats["path"]["status_counts"].items()
            if key != "success"
        },
        "retrievers": retriever_stats,
    }


def diagnostics_for_results(
    questions: Sequence[Mapping[str, str]], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ids_by_group = {
        "overall": {row["id"] for row in questions},
        "Fact Retrieval": {
            row["id"] for row in questions if row["question_type"] == "Fact Retrieval"
        },
        "Complex Reasoning": {
            row["id"]
            for row in questions
            if row["question_type"] == "Complex Reasoning"
        },
    }
    return {
        "schema_version": "phase4-v1",
        "scope": "coverage-only; no answer generation or correctness evaluation",
        "groups": {
            name: summarize_group(question_ids, results)
            for name, question_ids in ids_by_group.items()
        },
    }


def run_all_retrievers(
    resources: RetrievalResources, questions: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    vectors = resources.encode_queries([row["question"] for row in questions])
    vector_retriever = VectorRetriever(resources)
    neighborhood_retriever = NeighborhoodRetriever(resources)
    path_retriever = PathRetriever(resources)
    results: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        query_id, query = question["id"], question["question"]
        query_vector = vectors[index]
        link_started = time.perf_counter()
        links = resources.linker.link(query, query_vector)
        link_latency = (time.perf_counter() - link_started) * 1000
        vector_result = vector_retriever.retrieve(query_id, query, query_vector)
        neighborhood_result = neighborhood_retriever.retrieve(
            query_id, query, query_vector, links
        )
        path_result = path_retriever.retrieve(query_id, query, query_vector, links)
        for result in (neighborhood_result, path_result):
            result["diagnostics"]["entity_link_latency_ms"] = round(link_latency, 6)
            result["diagnostics"]["latency_ms"] = round(
                float(result["diagnostics"]["latency_ms"]) + link_latency, 6
            )
        results.extend((vector_result, neighborhood_result, path_result))
    return results


def render_phase4_report(
    diagnostics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    groups = diagnostics["groups"]
    group_rows = []
    for name in ("overall", "Fact Retrieval", "Complex Reasoning"):
        item = groups[name]
        group_rows.append(
            "| {name} | {count} | {link:.2%} | {multi:.2%} | {component:.2%} | "
            "{vector:.2%} | {neighborhood:.2%} | {path:.2%} |".format(
                name=name,
                count=item["query_count"],
                link=item["entity_link_success_rate"],
                multi=item["linked_at_least_two_rate"],
                component=item["same_component_rate"],
                vector=item["retrievers"]["vector"]["nonempty_rate"],
                neighborhood=item["neighborhood_nonempty_rate"],
                path=item["path_found_rate"],
            )
        )
    retriever_rows = []
    overall = groups["overall"]
    for name in RETRIEVER_NAMES:
        item = overall["retrievers"][name]
        latency = item["latency_ms"]
        retriever_rows.append(
            f"| {name} | {item['status_counts']} | {item['average_returned_chunks']:.4f} | "
            f"{latency['mean']:.3f} | {latency['p50']:.3f} | {latency['p95']:.3f} |"
        )
    embedding = manifest["embedding"]
    return f"""# 阶段 4：共享数据基础上的三个检索器

## 实现边界

本阶段在完全相同的 62 个 phase2 Chunk 和同一份 phase3 知识图谱上实现
`VectorRetriever`、`NeighborhoodRetriever` 和 `PathRetriever`。后两者分别是
LightRAG Local 风格的一跳邻域检索与 PathRAG 风格的关系路径检索课程简化版，
**不是** LightRAG 或 PathRAG 的完整复现。

本阶段未调用生成式 LLM、未生成答案、未评价答案正确率，未读取测试集，也没有
使用训练题中的 `answer`、`evidence`、`evidence_triple`。问题加载器只投影
`id`、`question`、`question_type`，其中题型仅用于本报告分组。

## 配置与索引

- 随机种子：{config['random_seed']}
- 共享 Chunk 预算：{config['top_k_chunks']}
- 最多种子实体：{config['max_seed_entities']}
- 实体语义阈值：{config['entity_similarity_threshold']}
- 邻域跳数：{config['neighborhood_hops']}
- 最大路径跳数：{config['max_path_hops']}；每实体对最多路径：{config['max_paths_per_pair']}
- Path 排除关系：{config['excluded_path_relations']}
- 嵌入模型：`{embedding['model_name']}`
- 模型 revision：`{embedding['model_revision']}`
- 权重 SHA-256：`{embedding['weights_sha256']}`
- 向量维度：{embedding['vector_dimension']}；pooling：{embedding['pooling']}；L2 normalize：{embedding['normalize']}
- 查询前缀：`{embedding['query_prefix']}`；文档不加前缀并保持原文。
- 编码环境：Transformers {embedding['transformers_version']}，PyTorch {embedding['torch_version']}，NumPy {embedding['numpy_version']}，NetworkX {embedding['networkx_version']}。

Vector 只对归一化 Chunk 向量进行余弦排序，不接触图。Neighborhood 先做规范名/
别名匹配，再用实体名称、别名和描述向量补充实体链接，展开一跳后仅从节点来源、
边来源与 `chunk_entity_map` 收集 Chunk。Path 在排除元数据边后的无向投影上搜索
不超过三跳的简单路径；失败时返回明确状态，不进行向量回退。

## 62 道训练问题覆盖率

“同一分量”按至少两个链接实体中存在一对在排除元数据边后的图中连通计算；这里
只测可检索覆盖，不使用答案或标准证据。

| 分组 | 问题数 | ≥1 实体 | ≥2 实体 | 同一分量 | Vector 非空 | 邻域非空 | 找到路径 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(group_rows)}

## 状态、返回量与耗时

| 检索器 | 状态计数 | 平均 Chunk 数 | 均值 ms | P50 ms | P95 ms |
|---|---|---:|---:|---:|---:|
{chr(10).join(retriever_rows)}

- Path 失败原因：`{overall['path_failure_reasons']}`
- 至少两个实体的问题中，同一分量比例：{overall['same_component_rate_among_multi_entity']:.2%}
- 所有图检索结果均记录 `vector_fallback_used=false`。

## 确定性与风险

- 结果结构哈希：`{manifest['result_structural_sha256']}`
- 诊断结构哈希：`{manifest['diagnostics_structural_sha256']}`
- 除实测耗时字段外，重复运行结果一致。
- phase3 图有较多孤立节点且关系过滤较保守，因此 Path 覆盖率可能显著低于向量
  覆盖率；本阶段如实保留失败状态，没有反向修改图谱。
- 实体链接仍可能受作品名/动物名 `Dandy Dick` 等同形别名影响；语义阈值和最多
  四个种子实体均固定在配置中，未根据答案或测试集调参。
- 下一步可以在这三个共享检索器之上评估检索证据，但本阶段不开始答案生成、
  伪标签或分类器训练。
"""
