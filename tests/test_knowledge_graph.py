from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from scripts.extract_knowledge_graph import (
    AliasResolver,
    build_networkx,
    canonical_json,
    evidence_matches,
    extraction_fingerprint,
    extract_chunks,
    load_config,
    merge_graph_data,
    normalize_ws,
    validate_extraction,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "kg_config.json"
ALIASES_PATH = ROOT / "configs" / "alias_overrides.json"
PROMPT_PATH = ROOT / "prompts" / "kg_extraction.txt"
CHUNKS_PATH = ROOT / "data" / "processed" / "phase2" / "chunks.jsonl"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase3"


def synthetic_chunk(chunk_id: str = "dd_act01_test") -> dict:
    return {
        "chunk_id": chunk_id,
        "corpus_name": "Novel-40700",
        "title": "Dandy Dick",
        "section_id": "act_1",
        "section_type": "play",
        "act": 1,
        "scene": None,
        "speakers": ["THE DEAN", "GEORGIANA"],
        "text": "THE DEAN. THE DEAN is Georgiana's brother. GEORGIANA. Yes, Gus.",
    }


def valid_payload(chunk_id: str = "dd_act01_test") -> dict:
    return {
        "entities": [
            {
                "name": "THE DEAN",
                "type": "CHARACTER",
                "description": "Georgiana's brother.",
                "aliases": ["Gus"],
            },
            {
                "name": "Georgiana",
                "type": "CHARACTER",
                "description": "The Dean's sister.",
                "aliases": [],
            },
        ],
        "relations": [
            {
                "source": "THE DEAN",
                "target": "Georgiana",
                "relation": "SIBLING_OF",
                "description": "They are siblings.",
                "evidence_quote": "THE DEAN is Georgiana's brother.",
                "source_chunk_id": chunk_id,
                "confidence": 0.99,
            }
        ],
        "no_fact_reason": None,
    }


class KnowledgeGraphCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        cls.aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
        cls.prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def test_evidence_matching_ignores_only_whitespace(self) -> None:
        text = "The Dean\n owns   a horse named Dandy Dick."
        self.assertTrue(evidence_matches("The Dean owns a horse named Dandy Dick.", text))
        self.assertFalse(evidence_matches("The Dean sold Dandy Dick.", text))
        self.assertEqual(normalize_ws(" a\n b\t c "), "a b c")

    def test_extraction_schema_and_evidence_are_validated(self) -> None:
        chunk = synthetic_chunk()
        parsed, warnings = validate_extraction(valid_payload(), chunk, self.config)
        self.assertEqual(len(parsed["entities"]), 2)
        self.assertEqual(len(parsed["relations"]), 1)
        self.assertEqual(warnings, [])
        broken = valid_payload()
        broken["relations"][0]["evidence_quote"] = "invented sentence"
        parsed, warnings = validate_extraction(broken, chunk, self.config)
        self.assertEqual(parsed["relations"], [])
        self.assertIn("relation[0]_evidence_mismatch", warnings)

    def test_required_aliases_and_ambiguous_alias(self) -> None:
        resolver = AliasResolver(self.aliases)
        for alias in ("THE DEAN", "Augustin", "Gus", "Dr. Jedd"):
            self.assertEqual(resolver.resolve(alias, "CHARACTER")[0], "Augustin Jedd")
        self.assertEqual(
            resolver.resolve("George Tidd", "CHARACTER")[0], "Georgiana Tidman"
        )
        self.assertEqual(
            resolver.resolve("Tris", "CHARACTER")[0], "Sir Tristram Mardon"
        )
        self.assertEqual(resolver.resolve("Miss Jedd", "CHARACTER")[0], None)

    def test_alias_merge_and_symmetric_relation_deduplication(self) -> None:
        first_chunk = synthetic_chunk("dd_act01_test1")
        second_chunk = synthetic_chunk("dd_act01_test2")
        second_chunk["text"] = "GEORGIANA. Georgiana says Gus is her brother."
        first = valid_payload(first_chunk["chunk_id"])
        second = {
            "entities": [
                {
                    "name": "Georgiana Tidman",
                    "type": "CHARACTER",
                    "description": "Gus's sister.",
                    "aliases": [],
                },
                {
                    "name": "Gus",
                    "type": "CHARACTER",
                    "description": "Georgiana's brother.",
                    "aliases": [],
                },
            ],
            "relations": [
                {
                    "source": "Georgiana Tidman",
                    "target": "Gus",
                    "relation": "SIBLING_OF",
                    "description": "They are siblings.",
                    "evidence_quote": "Georgiana says Gus is her brother.",
                    "source_chunk_id": second_chunk["chunk_id"],
                    "confidence": 0.98,
                }
            ],
            "no_fact_reason": None,
        }
        parsed_first, _ = validate_extraction(first, first_chunk, self.config)
        parsed_second, _ = validate_extraction(second, second_chunk, self.config)
        outcomes = [
            {
                "chunk_id": first_chunk["chunk_id"],
                "outcome": "success",
                "validated_output": parsed_first,
            },
            {
                "chunk_id": second_chunk["chunk_id"],
                "outcome": "success",
                "validated_output": parsed_second,
            },
        ]
        merged = merge_graph_data(
            [first_chunk, second_chunk], outcomes, self.config, self.aliases
        )
        names = {node["name"] for node in merged["nodes"]}
        self.assertEqual(names, {"Augustin Jedd", "Georgiana Tidman"})
        self.assertEqual(len(merged["edges"]), 1)
        self.assertFalse(merged["edges"][0]["directed"])
        self.assertEqual(len(merged["edges"][0]["evidence"]), 2)

    def test_character_and_person_are_not_merged(self) -> None:
        resolver = AliasResolver(self.aliases)
        canonical, method = resolver.resolve("THE DEAN", "PERSON")
        self.assertIsNone(canonical)
        self.assertEqual(method, "type_conflict_omitted")

    def test_graphml_round_trip_and_json_string_arrays(self) -> None:
        nodes = [
            {
                "node_id": "n_a",
                "name": "A",
                "type": "CHARACTER",
                "description": "A character.",
                "descriptions": ["A character."],
                "aliases": ["Alias A"],
                "source_chunk_ids": ["c1"],
            },
            {
                "node_id": "n_b",
                "name": "B",
                "type": "CHARACTER",
                "description": "B character.",
                "descriptions": ["B character."],
                "aliases": [],
                "source_chunk_ids": ["c1"],
            },
        ]
        edges = [
            {
                "edge_id": "e_ab",
                "source_id": "n_a",
                "target_id": "n_b",
                "source": "A",
                "target": "B",
                "relation": "FRIEND_OF",
                "directed": False,
                "description": "friends",
                "descriptions": ["friends"],
                "evidence_quote": "A and B are friends.",
                "source_chunk_id": "c1",
                "confidence": 0.9,
                "evidence": [
                    {
                        "evidence_quote": "A and B are friends.",
                        "source_chunk_id": "c1",
                        "confidence": 0.9,
                    }
                ],
                "source_chunk_ids": ["c1"],
            }
        ]
        graph = build_networkx(nodes, edges)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "graph.graphml"
            nx.write_graphml(graph, path)
            loaded = nx.read_graphml(path)
        self.assertEqual(loaded.number_of_nodes(), 2)
        self.assertEqual(loaded.number_of_edges(), 1)
        self.assertEqual(json.loads(loaded.nodes["n_a"]["aliases"]), ["Alias A"])

    def test_valid_cache_is_reused_offline(self) -> None:
        chunk = synthetic_chunk()
        parsed, warnings = validate_extraction(valid_payload(), chunk, self.config)
        model = "fixture-model"
        fingerprint = extraction_fingerprint(self.prompt, self.config, model)
        cached = {
            "cache_version": "phase3-cache-v1",
            "chunk_id": chunk["chunk_id"],
            "input_sha256": __import__(
                "scripts.extract_knowledge_graph", fromlist=["sha256_text"]
            ).sha256_text(chunk["text"]),
            "extraction_fingerprint": fingerprint,
            "model": model,
            "outcome": "success",
            "validated_output": parsed,
            "validation_warnings": warnings,
            "request_audit": {
                "attempt_count": 1,
                "attempts": [],
                "total_elapsed_seconds": 0.1,
                "token_usage": {"total_tokens": 10},
            },
            "raw_response": {},
        }
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            (cache_dir / f"{chunk['chunk_id']}.json").write_text(
                canonical_json(cached, pretty=True), encoding="utf-8"
            )
            outcomes, run = extract_chunks(
                [chunk],
                self.config,
                self.prompt,
                {
                    "LLM_API_KEY": "",
                    "LLM_BASE_URL": "",
                    "LLM_MODEL": model,
                },
                cache_dir,
                offline=True,
                retry_failures=False,
            )
        self.assertEqual(outcomes[0]["outcome"], "success")
        self.assertEqual(run["api_calls_this_run"], 0)
        self.assertEqual(run["cache_hits_this_run"], 1)


class KnowledgeGraphArtifactTests(unittest.TestCase):
    def test_phase3_artifacts_and_quality_report(self) -> None:
        required = {
            "nodes.jsonl",
            "edges.jsonl",
            "graph.json",
            "graph.graphml",
            "chunk_entity_map.json",
            "alias_map.json",
            "unresolved_aliases.json",
            "extraction_failures.jsonl",
            "llm_usage.json",
            "kg_manifest.json",
            "quality_report.json",
        }
        self.assertTrue(OUTPUT_DIR.is_dir(), "run phase3 extraction first")
        self.assertTrue(required <= {path.name for path in OUTPUT_DIR.iterdir()})
        manifest = json.loads((OUTPUT_DIR / "kg_manifest.json").read_text())
        quality = json.loads((OUTPUT_DIR / "quality_report.json").read_text())
        self.assertEqual(manifest["input_chunk_count"], 62)
        self.assertEqual(quality["status"], "pass")
        self.assertTrue(all(item["passed"] for item in quality["checks"]))
        self.assertEqual(
            quality["postprocess_digest_first"], quality["postprocess_digest_second"]
        )

    def test_all_quotes_match_phase2_chunks_and_no_all_node(self) -> None:
        chunks = {
            row["chunk_id"]: row
            for row in (
                json.loads(line)
                for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
            )
        }
        nodes = [
            json.loads(line)
            for line in (OUTPUT_DIR / "nodes.jsonl").read_text().splitlines()
        ]
        edges = [
            json.loads(line)
            for line in (OUTPUT_DIR / "edges.jsonl").read_text().splitlines()
        ]
        self.assertNotIn("ALL", {node["name"] for node in nodes})
        for edge in edges:
            for evidence in edge["evidence"]:
                self.assertIn(evidence["source_chunk_id"], chunks)
                self.assertTrue(
                    evidence_matches(
                        evidence["evidence_quote"],
                        chunks[evidence["source_chunk_id"]]["text"],
                    )
                )

    def test_graph_files_reload(self) -> None:
        graph_json = json.loads((OUTPUT_DIR / "graph.json").read_text())
        graphml = nx.read_graphml(OUTPUT_DIR / "graph.graphml")
        self.assertEqual(len(graph_json["nodes"]), graphml.number_of_nodes())
        self.assertEqual(len(graph_json["edges"]), graphml.number_of_edges())

    def test_manual_review_and_core_semantics(self) -> None:
        review = json.loads(
            (ROOT / "data" / "interim" / "phase3" / "manual_review.json").read_text()
        )
        nodes = [
            json.loads(line)
            for line in (OUTPUT_DIR / "nodes.jsonl").read_text().splitlines()
        ]
        edges = [
            json.loads(line)
            for line in (OUTPUT_DIR / "edges.jsonl").read_text().splitlines()
        ]
        self.assertGreaterEqual(len(review["sample"]), 30)
        self.assertTrue(all(row["manual_verdict"] is True for row in review["sample"]))
        self.assertEqual(
            {node["type"] for node in nodes if node["name"] == "Dandy Dick"},
            {"ANIMAL"},
        )
        self.assertNotIn("INTERACTS_WITH", {edge["relation"] for edge in edges})

    def test_offline_rebuild_is_byte_deterministic(self) -> None:
        paths = [
            *(OUTPUT_DIR / name for name in (
                "nodes.jsonl",
                "edges.jsonl",
                "graph.json",
                "graph.graphml",
                "chunk_entity_map.json",
                "alias_map.json",
                "unresolved_aliases.json",
                "extraction_failures.jsonl",
                "llm_usage.json",
                "kg_manifest.json",
                "quality_report.json",
            )),
            ROOT / "data" / "interim" / "phase3" / "manual_review.json",
            ROOT / "reports" / "phase3_knowledge_graph.md",
        ]

        def digests() -> dict[str, str]:
            return {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            }

        before = digests()
        completed = subprocess.run(
            [sys.executable, "scripts/extract_knowledge_graph.py", "--offline"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout[-2000:])
        self.assertEqual(before, digests())


if __name__ == "__main__":
    unittest.main()
