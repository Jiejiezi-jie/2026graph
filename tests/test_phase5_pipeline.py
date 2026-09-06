from __future__ import annotations

import collections
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.phase5_pipeline import (
    GENERATION_PAYLOAD_FIELDS,
    RETRIEVERS,
    Phase5Error,
    build_generation_messages,
    call_generation_with_cache,
    canonical_json,
    choose_router_label,
    evidence_coverage,
    generation_cache_path,
    load_config,
    load_generation_questions,
    load_and_validate_retrieval_results,
    load_chunks,
    normalize_official_classification,
    normalize_for_evidence,
    sha256_file,
    sha256_text,
    token_recall,
    validate_frozen_inputs,
    validate_generation_payload,
    validate_training_schema,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "phase5_generation_evaluation.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase5"


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def call_once(self, messages, *, strict_object):
        self.calls += 1
        response = {
            "choices": [
                {
                    "message": {"content": json.dumps(self.payload)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return response, 0.01


class Phase5CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.paths = load_config(ROOT, CONFIG_PATH)

    def test_frozen_input_hashes_and_complete_retrieval_matrix(self) -> None:
        hashes = validate_frozen_inputs(self.config, self.paths)
        self.assertEqual(set(hashes), {"train_questions", "chunks", "retrieval_results"})
        validate_training_schema(self.paths["train_questions"], 62)
        questions = load_generation_questions(self.paths["train_questions"])
        self.assertEqual(set(questions[0]), {"id", "question"})
        _chunks, chunk_by_id = load_chunks(self.paths["chunks"], 62)
        rows = load_and_validate_retrieval_results(
            self.paths["retrieval_results"], questions, chunk_by_id, 186, 5
        )
        self.assertEqual(len(rows), 186)
        self.assertEqual(
            collections.Counter(row["retriever"] for row in rows),
            {"vector": 62, "neighborhood": 62, "path": 62},
        )
        self.assertEqual(
            sum(row["retriever"] == "path" and row["status"] != "success" for row in rows),
            10,
        )

    def test_generation_prompt_contains_no_supervision(self) -> None:
        result = {
            "retriever": "neighborhood",
            "retrieved_chunks": [{"chunk_id": "c1", "rank": 1, "score": 1.0}],
            "linked_entities": [{"node_id": "alice", "name": "Alice"}],
            "retrieved_nodes": [{"node_id": "alice", "name": "Alice"}],
            "retrieved_edges": [],
            "paths": [],
        }
        messages, chunk_ids, context = build_generation_messages(
            "Grounded system prompt",
            "Where is Alice?",
            result,
            {"c1": "Alice is at the station."},
            5,
        )
        serialized = canonical_json(messages)
        self.assertEqual(chunk_ids, ["c1"])
        self.assertGreater(len(context), 20)
        self.assertIn("Where is Alice?", serialized)
        self.assertIn("Alice is at the station.", serialized)
        self.assertNotIn("SECRET_STANDARD_ANSWER", serialized)
        self.assertNotIn("SECRET_STANDARD_EVIDENCE", serialized)

    def test_strict_payload_and_invalid_citation_audit(self) -> None:
        payload, invalid = validate_generation_payload(
            {
                "answer": "Alice is at the station.",
                "citations": ["c1", "invented", "c1"],
                "insufficient_evidence": False,
            },
            ["c1"],
        )
        self.assertEqual(set(payload), GENERATION_PAYLOAD_FIELDS)
        self.assertEqual(payload["citations"], ["c1", "invented"])
        self.assertEqual(invalid, ["invented"])
        with self.assertRaises(Phase5Error):
            validate_generation_payload(
                {"answer": "maybe", "citations": [], "insufficient_evidence": True},
                ["c1"],
            )

    def test_generation_cache_replays_offline_without_new_call(self) -> None:
        messages = [
            {"role": "system", "content": "Return JSON"},
            {"role": "user", "content": "Question and c1"},
        ]
        payload = {
            "answer": "Grounded answer.",
            "citations": ["c1"],
            "insufficient_evidence": False,
        }
        generation_config = self.config["generation"]
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            client = FakeClient(payload)
            first, first_stats = call_generation_with_cache(
                client=client,
                messages=messages,
                query_id="q1",
                retriever="vector",
                chunk_ids=["c1"],
                cache_dir=cache_dir,
                generation_config=generation_config,
                model="fake-model",
                offline=False,
                retry_failures=False,
            )
            second, second_stats = call_generation_with_cache(
                client=None,
                messages=messages,
                query_id="q1",
                retriever="vector",
                chunk_ids=["c1"],
                cache_dir=cache_dir,
                generation_config=generation_config,
                model="fake-model",
                offline=True,
                retry_failures=False,
            )
            self.assertEqual(client.calls, 1)
            self.assertEqual(first_stats["api_calls"], 1)
            self.assertEqual(second_stats["api_calls"], 0)
            self.assertEqual(second_stats["cache_hits"], 1)
            self.assertEqual(first, second)
            prompt_hash = sha256_text(canonical_json(messages))
            self.assertTrue(
                generation_cache_path(cache_dir, "q1", "vector", prompt_hash).is_file()
            )

    def test_evidence_coverage_normalization_and_token_threshold(self) -> None:
        self.assertEqual(normalize_for_evidence("  DANDY—Dick! "), "dandy dick")
        self.assertGreaterEqual(
            token_recall("Alice's horse Dandy Dick", "Alice owns horse Dandy Dick"),
            0.8,
        )
        score, details = evidence_coverage(
            ["Alice's horse Dandy Dick", "Bob was arrested"],
            ["c1"],
            {"c1": "Alice owns horse Dandy Dick."},
            0.8,
        )
        self.assertEqual(score, 0.5)
        self.assertTrue(details[0]["covered"])
        self.assertFalse(details[1]["covered"])

    def test_label_rule_prefers_simplest_acceptable_then_best_available(self) -> None:
        difficulty = {"vector": 0, "neighborhood": 1, "path": 2}
        methods = {
            "vector": {"acceptable": True, "answer_correctness": 0.8, "evidence_coverage": 0.5, "total_tokens": 50},
            "neighborhood": {"acceptable": True, "answer_correctness": 0.9, "evidence_coverage": 1.0, "total_tokens": 40},
            "path": {"acceptable": False, "answer_correctness": 0.2, "evidence_coverage": 0.2, "total_tokens": 10},
        }
        self.assertEqual(choose_router_label(methods, difficulty)[0:2], ("vector", "acceptable"))
        for row in methods.values():
            row["acceptable"] = False
        methods["vector"].update(answer_correctness=0.7, evidence_coverage=1.0)
        methods["neighborhood"].update(answer_correctness=0.7, evidence_coverage=1.0, total_tokens=40)
        self.assertEqual(
            choose_router_label(methods, difficulty)[0:2],
            ("neighborhood", "best_available"),
        )

    def test_official_classification_compatibility_preserves_counts(self) -> None:
        normalized = normalize_official_classification(
            {
                "classification": {
                    "true_positives": ["supported statement"],
                    "false_positives": 0,
                    "false_negatives": [
                        {"claim": "missing statement", "reason": "not answered"}
                    ],
                }
            }
        )
        self.assertEqual({key: len(value) for key, value in normalized.items()}, {
            "TP": 1,
            "FP": 0,
            "FN": 1,
        })


@unittest.skipUnless(
    (OUTPUT_DIR / "phase5_manifest.json").is_file(),
    "phase-five API artifacts have not been generated yet",
)
class Phase5ArtifactTests(unittest.TestCase):
    def load_jsonl(self, name: str) -> list[dict]:
        return [
            json.loads(line)
            for line in (OUTPUT_DIR / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_artifact_counts_labels_and_no_supervision_leak(self) -> None:
        answers = self.load_jsonl("train_answers.jsonl")
        scores = self.load_jsonl("train_method_scores.jsonl")
        labels = self.load_jsonl("train_router_labels.jsonl")
        self.assertEqual(len(answers), 186)
        self.assertEqual(len(scores), 186)
        self.assertEqual(len(labels), 62)
        self.assertEqual(len({row["query_id"] for row in labels}), 62)
        self.assertEqual(sum(row["generation"]["llm_called"] for row in answers), 176)
        self.assertEqual(
            sum(row["generation_status"] == "retrieval_failure" for row in answers), 10
        )
        for row in answers:
            self.assertFalse({"evidence", "evidence_triple", "question_type"} & set(row))
            self.assertLessEqual(len(row["context_chunk_ids"]), 5)

    def test_manifest_hashes_offline_replay_and_protocol_flags(self) -> None:
        manifest = json.loads(
            (OUTPUT_DIR / "phase5_manifest.json").read_text(encoding="utf-8")
        )
        config, paths = load_config(ROOT, CONFIG_PATH)
        self.assertEqual(validate_frozen_inputs(config, paths), {
            key: item["sha256"] for key, item in config["inputs"].items()
        })
        self.assertEqual(manifest["counts"], {
            "questions": 62,
            "train_answers": 186,
            "method_scores": 186,
            "router_labels": 62,
        })
        self.assertEqual(manifest["generation"]["offline_replay_api_calls"], 0)
        self.assertEqual(manifest["evaluation"]["offline_replay_api_calls"], 0)
        self.assertFalse(manifest["test_set_accessed"])
        self.assertFalse(manifest["retrievers_rerun"])
        self.assertFalse(manifest["router_trained"])
        self.assertFalse(manifest["standard_answers_in_generation_prompt"])
        for key, item in manifest["outputs"].items():
            self.assertEqual(sha256_file(ROOT / item["path"]), item["sha256"], key)

    def test_generation_caches_are_prompt_bound_and_secret_free(self) -> None:
        cache_dir = ROOT / "data" / "interim" / "phase5" / "generation_responses"
        caches = list(cache_dir.glob("*.json"))
        self.assertEqual(len(caches), 176)
        for path in caches:
            cache = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                sha256_text(canonical_json(cache["request_messages"])),
                cache["prompt_sha256"],
            )
            serialized = canonical_json(cache).casefold()
            self.assertNotIn("llm_api_key", serialized)
            self.assertNotIn("authorization", serialized)

    def test_non_timing_outputs_repeat_exactly_offline(self) -> None:
        paths = [
            OUTPUT_DIR / "train_answers.jsonl",
            OUTPUT_DIR / "train_method_scores.jsonl",
            OUTPUT_DIR / "train_router_labels.jsonl",
            OUTPUT_DIR / "method_summary.json",
            OUTPUT_DIR / "phase5_manifest.json",
            ROOT / "reports" / "phase5_generation_and_labels.md",
        ]
        before = {path: path.read_bytes() for path in paths}
        for script in ("generate_train_answers.py", "evaluate_and_build_labels.py"):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--offline"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
        after = {path: path.read_bytes() for path in paths}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
