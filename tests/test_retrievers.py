from __future__ import annotations

import collections
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from src.phase4_retrieval import (
    FORBIDDEN_RESULT_KEYS,
    REQUIRED_RESULT_KEYS,
    NeighborhoodRetriever,
    PathRetriever,
    RetrievalResources,
    VectorRetriever,
    structural_sha256,
    strip_timing_fields,
    validate_result,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE4_DIR = ROOT / "data" / "processed" / "phase4"
INDEX_DIR = ROOT / "data" / "interim" / "phase4"


class StaticEncoder:
    dimension = 2

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def link(node_id: str, name: str, score: float = 1.0) -> dict:
    return {
        "node_id": node_id,
        "name": name,
        "type": "CHARACTER" if node_id != "actor" else "PERSON",
        "score": score,
        "semantic_score": score,
        "method": "canonical",
        "matched_term": name,
    }


def toy_resources() -> RetrievalResources:
    chunks = [
        {"chunk_id": "c1", "text": "Alice owns a horse.", "section_id": "act_1"},
        {"chunk_id": "c2", "text": "Alice knows Bob.", "section_id": "act_1"},
        {"chunk_id": "c3", "text": "Bob helps Alice.", "section_id": "act_2"},
        {"chunk_id": "c4", "text": "An actor portrays Alice.", "section_id": "history"},
        {"chunk_id": "c5", "text": "Unrelated vector-only text.", "section_id": "act_3"},
    ]
    nodes = [
        {
            "node_id": "alice",
            "name": "Alice",
            "type": "CHARACTER",
            "aliases": ["Al"],
            "description": "A character.",
            "source_chunk_ids": ["c1"],
        },
        {
            "node_id": "bob",
            "name": "Bob",
            "type": "CHARACTER",
            "aliases": [],
            "description": "Alice's friend.",
            "source_chunk_ids": ["c2", "c3"],
        },
        {
            "node_id": "actor",
            "name": "Actor Person",
            "type": "PERSON",
            "aliases": [],
            "description": "Portrays Alice.",
            "source_chunk_ids": ["c4"],
        },
    ]
    edges = [
        {
            "edge_id": "e_friend",
            "source_id": "alice",
            "target_id": "bob",
            "source": "Alice",
            "target": "Bob",
            "relation": "FRIEND_OF",
            "directed": False,
            "source_chunk_ids": ["c2"],
        },
        {
            "edge_id": "e_opposes",
            "source_id": "alice",
            "target_id": "bob",
            "source": "Alice",
            "target": "Bob",
            "relation": "OPPOSES",
            "directed": True,
            "source_chunk_ids": ["c3"],
        },
        {
            "edge_id": "e_portrayed",
            "source_id": "alice",
            "target_id": "actor",
            "source": "Alice",
            "target": "Actor Person",
            "relation": "PORTRAYED_BY",
            "directed": True,
            "source_chunk_ids": ["c4"],
        },
    ]
    chunk_map = {
        "chunks": {
            "c1": {"node_refs": [{"node_id": "alice"}]},
            "c2": {"node_refs": [{"node_id": "alice"}, {"node_id": "bob"}]},
            "c3": {"node_refs": [{"node_id": "bob"}]},
            "c4": {"node_refs": [{"node_id": "actor"}]},
            "c5": {"node_refs": []},
        }
    }
    chunk_vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.99, 0.14106736],
        ],
        dtype=np.float32,
    )
    entity_vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    config = {
        "top_k_chunks": 2,
        "max_seed_entities": 4,
        "entity_similarity_threshold": 0.55,
        "neighborhood_hops": 1,
        "max_path_hops": 3,
        "max_paths_per_pair": 3,
        "excluded_path_relations": ["AUTHORED_BY", "PORTRAYED_BY"],
        "neighborhood_scoring": {
            "entity_weight": 0.6,
            "chunk_similarity_weight": 0.4,
            "hop_decay": 0.85,
        },
        "path_scoring": {
            "endpoint_weight": 0.45,
            "length_weight": 0.2,
            "chunk_similarity_weight": 0.35,
        },
    }
    return RetrievalResources(
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        chunk_entity_map=chunk_map,
        chunk_embeddings=chunk_vectors,
        entity_embeddings=entity_vectors,
        encoder=StaticEncoder(),
        config=config,
    )


class RetrieverCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resources = toy_resources()
        self.query_vector = np.asarray([1.0, 0.0], dtype=np.float32)

    def assert_unified(self, result: dict) -> None:
        self.assertEqual(set(result), REQUIRED_RESULT_KEYS)
        self.assertFalse(FORBIDDEN_RESULT_KEYS & set(result))
        validate_result(result, self.resources.config["top_k_chunks"])

    def test_vector_is_non_graph_and_obeys_top_k(self) -> None:
        result = VectorRetriever(self.resources).retrieve(
            "q1", "Alice", self.query_vector
        )
        self.assert_unified(result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            [row["chunk_id"] for row in result["retrieved_chunks"]], ["c1", "c5"]
        )
        self.assertEqual(result["linked_entities"], [])
        self.assertEqual(result["retrieved_nodes"], [])
        self.assertEqual(result["retrieved_edges"], [])
        self.assertEqual(result["paths"], [])

    def test_neighborhood_returns_only_graph_anchored_candidates(self) -> None:
        result = NeighborhoodRetriever(self.resources).retrieve(
            "q1", "Alice", self.query_vector, [link("alice", "Alice")]
        )
        self.assert_unified(result)
        self.assertEqual(result["status"], "success")
        self.assertLessEqual(len(result["retrieved_chunks"]), 2)
        self.assertIn("bob", {row["node_id"] for row in result["retrieved_nodes"]})
        self.assertFalse(result["diagnostics"]["vector_fallback_used"])
        self.assertNotIn(
            "c5", {row["chunk_id"] for row in result["retrieved_chunks"]}
        )
        self.assertLess(result["diagnostics"]["candidate_chunk_count"], 5)

    def test_path_deduplicates_parallel_edges_and_excludes_metadata(self) -> None:
        result = PathRetriever(self.resources).retrieve(
            "q1",
            "Alice and Bob",
            self.query_vector,
            [link("alice", "Alice"), link("bob", "Bob", 0.9)],
        )
        self.assert_unified(result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["paths"]), 1)
        self.assertEqual(result["paths"][0]["node_ids"], ["alice", "bob"])
        self.assertNotIn(
            "PORTRAYED_BY",
            {edge["relation"] for edge in result["retrieved_edges"]},
        )
        self.assertFalse(result["diagnostics"]["vector_fallback_used"])

    def test_graph_failures_do_not_fall_back_to_vector(self) -> None:
        path = PathRetriever(self.resources)
        insufficient = path.retrieve(
            "q1", "Alice", self.query_vector, [link("alice", "Alice")]
        )
        disconnected = path.retrieve(
            "q2",
            "Alice and the actor",
            self.query_vector,
            [link("alice", "Alice"), link("actor", "Actor Person")],
        )
        for result, expected in (
            (insufficient, "insufficient_entities"),
            (disconnected, "disconnected"),
        ):
            self.assert_unified(result)
            self.assertEqual(result["status"], expected)
            self.assertEqual(result["retrieved_chunks"], [])
            self.assertFalse(result["diagnostics"]["vector_fallback_used"])

    def test_repeated_toy_results_match_except_timing(self) -> None:
        retriever = NeighborhoodRetriever(self.resources)
        first = retriever.retrieve(
            "q1", "Alice", self.query_vector, [link("alice", "Alice")]
        )
        second = retriever.retrieve(
            "q1", "Alice", self.query_vector, [link("alice", "Alice")]
        )
        self.assertEqual(strip_timing_fields(first), strip_timing_fields(second))


class Phase4ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results_path = PHASE4_DIR / "train_retrieval_results.jsonl"
        cls.diagnostics_path = PHASE4_DIR / "retrieval_diagnostics.json"
        cls.manifest_path = PHASE4_DIR / "retrieval_manifest.json"

    def load_results(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_complete_artifacts_and_common_budget(self) -> None:
        results = self.load_results()
        self.assertEqual(len(results), 186)
        self.assertEqual(
            collections.Counter(row["retriever"] for row in results),
            {"vector": 62, "neighborhood": 62, "path": 62},
        )
        for result in results:
            self.assertEqual(set(result), REQUIRED_RESULT_KEYS)
            self.assertFalse(FORBIDDEN_RESULT_KEYS & set(result))
            validate_result(result, 5)
        self.assertLessEqual(
            max(len(row["retrieved_chunks"]) for row in results), 5
        )

    def test_path_limits_filters_and_no_fallback(self) -> None:
        for result in self.load_results():
            if result["retriever"] != "path":
                continue
            relation_types = {
                relation for path in result["paths"] for relation in path["relations"]
            }
            self.assertFalse({"AUTHORED_BY", "PORTRAYED_BY"} & relation_types)
            pair_counts = collections.Counter(
                (path["source_id"], path["target_id"]) for path in result["paths"]
            )
            self.assertTrue(all(count <= 3 for count in pair_counts.values()))
            node_sequences = [tuple(path["node_ids"]) for path in result["paths"]]
            self.assertEqual(len(node_sequences), len(set(node_sequences)))
            if result["status"] in {
                "insufficient_entities",
                "disconnected",
                "no_path",
            }:
                self.assertEqual(result["retrieved_chunks"], [])
            self.assertFalse(result["diagnostics"]["vector_fallback_used"])

    def test_embedding_index_metadata_and_normalization(self) -> None:
        metadata = json.loads(
            (INDEX_DIR / "index_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["embedding"]["model_name"], "BAAI/bge-small-en-v1.5")
        self.assertEqual(metadata["embedding"]["vector_dimension"], 384)
        self.assertEqual(metadata["chunk_count"], 62)
        self.assertEqual(metadata["entity_count"], 124)
        chunks = np.load(INDEX_DIR / "chunk_embeddings.npy", allow_pickle=False)
        entities = np.load(INDEX_DIR / "entity_embeddings.npy", allow_pickle=False)
        self.assertEqual(chunks.shape, (62, 384))
        self.assertEqual(entities.shape, (124, 384))
        self.assertTrue(np.allclose(np.linalg.norm(chunks, axis=1), 1.0, atol=2e-5))
        self.assertTrue(np.allclose(np.linalg.norm(entities, axis=1), 1.0, atol=2e-5))

    def test_manifest_structural_hashes_match_artifacts(self) -> None:
        results = self.load_results()
        diagnostics = json.loads(self.diagnostics_path.read_text(encoding="utf-8"))
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["result_structural_sha256"], structural_sha256(results)
        )
        self.assertEqual(
            manifest["diagnostics_structural_sha256"],
            structural_sha256(diagnostics),
        )
        self.assertEqual(manifest["generative_llm_calls"], 0)
        self.assertFalse(manifest["answer_generation_performed"])

    def test_full_diagnostics_repeat_except_timing(self) -> None:
        rewritten_paths = [
            self.results_path,
            self.diagnostics_path,
            self.manifest_path,
            ROOT / "reports" / "phase4_retrievers.md",
        ]
        snapshots = {path: path.read_bytes() for path in rewritten_paths}
        before_results = strip_timing_fields(self.load_results())
        before_diagnostics = strip_timing_fields(
            json.loads(self.diagnostics_path.read_text(encoding="utf-8"))
        )
        try:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_retrieval_diagnostics.py")],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            after_results = strip_timing_fields(self.load_results())
            after_diagnostics = strip_timing_fields(
                json.loads(self.diagnostics_path.read_text(encoding="utf-8"))
            )
            self.assertEqual(before_results, after_results)
            self.assertEqual(before_diagnostics, after_diagnostics)
        finally:
            for path, payload in snapshots.items():
                path.write_bytes(payload)


if __name__ == "__main__":
    unittest.main()
