import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from src.official_analysis import choose_official_silver
from src.official_backends.lightrag_backend import LightRAGBackend
from src.official_backends.model_client import (
    Generation,
    OpenAICompatibleChatClient,
    OpenAICompatibleEmbeddingClient,
    UsageSnapshot,
)
from src.official_backends.pathrag_backend import PathRAGBackend, parse_pathrag_context
from src.official_backends.vector_backend import VectorRAGBackend, chunk_by_token_window
from src.official_data import p0_subset, stratified_sample_and_split


class FakeEmbedding:
    model_path = Path("fake-embedding")
    max_length = 512
    dimension = 2
    batch_size = 2

    def __init__(self):
        self.lock = threading.Lock()

    async def embed(self, texts, **kwargs):
        vectors = []
        for text in texts:
            vector = np.array(
                [text.lower().count("cancer") + 1, text.lower().count("skin") + 1],
                dtype=np.float32,
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)


class FakeLLM:
    model_name = "fake-llm"

    def __init__(self):
        self.usage = UsageSnapshot()
        self.lock = threading.Lock()

    async def generate(self, prompt, system_prompt=None, **kwargs):
        self.usage = UsageSnapshot(
            self.usage.input_tokens + 10, self.usage.output_tokens + 2, self.usage.calls + 1
        )
        return Generation("grounded answer", 10, 2)

    async def complete(self, prompt, system_prompt=None, **kwargs):
        return (await self.generate(prompt, system_prompt=system_prompt, **kwargs)).text

    def snapshot(self):
        return self.usage


class OfficialCoreTests(unittest.IsolatedAsyncioTestCase):
    def test_frozen_stratified_split(self):
        questions = []
        for kind in ("Fact Retrieval", "Complex Reasoning"):
            for index in range(100):
                questions.append(
                    {
                        "id": f"{kind}-{index}",
                        "question": f"Question {index}",
                        "question_type": kind,
                    }
                )
        first = stratified_sample_and_split(
            questions, ["Fact Retrieval", "Complex Reasoning"], 60, seed=42
        )
        second = stratified_sample_and_split(
            questions, ["Fact Retrieval", "Complex Reasoning"], 60, seed=42
        )
        self.assertEqual(first, second)
        self.assertEqual(len(p0_subset(first, ["Fact Retrieval", "Complex Reasoning"], 5)), 10)
        self.assertEqual(sum(row["split"] == "test" for row in first), 24)

    def test_token_chunking(self):
        chunks = chunk_by_token_window("skin cancer " * 100, size=30, overlap=5)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk for chunk in chunks))

    async def test_vector_backend_schema_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = VectorRAGBackend(
                directory,
                FakeLLM(),
                FakeEmbedding(),
                chunk_tokens=20,
                chunk_overlap_tokens=2,
                top_k=2,
                max_context_tokens=100,
            )
            index = await backend.index("skin cancer evidence. " * 50)
            result = await backend.query("Which skin cancer?", "q-1")
            self.assertFalse(index["cached"])
            self.assertEqual(result["method"], "vector")
            self.assertEqual(result["question_id"], "q-1")
            self.assertEqual(result["llm_calls"], 1)
            self.assertTrue(result["contexts"])
            self.assertEqual(result["answer"], "grounded answer")

    async def test_lightrag_v157_adapter_initializes(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LightRAGBackend(directory, FakeLLM(), FakeEmbedding())
            await backend._initialize()
            self.assertIsNotNone(backend.rag)
            await backend.close()

    async def test_pathrag_adapter_initializes_with_locked_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = PathRAGBackend(
                Path(__file__).parents[1] / "third_party" / "PathRAG",
                directory,
                FakeLLM(),
                FakeEmbedding(),
            )
            await backend._initialize()
            self.assertIsNotNone(backend.rag)
            await backend.close()

    async def test_openai_compatible_chat_preserves_structured_output(self):
        client = OpenAICompatibleChatClient(
            "https://example.test/v1",
            "test-key",
            "test-chat",
            max_new_tokens=32,
            max_callback_new_tokens=128,
            max_retries=0,
        )
        payloads = []

        def fake_request(payload):
            payloads.append(payload)
            return {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }

        client._request_sync = fake_request
        response = await client.complete("Return JSON", keyword_extraction=True)
        self.assertEqual(response, '{"ok": true}')
        self.assertEqual(payloads[0]["response_format"], {"type": "json_object"})
        self.assertEqual(payloads[0]["max_tokens"], 128)
        self.assertEqual(client.snapshot(), UsageSnapshot(12, 4, 1))

    async def test_openai_compatible_embedding_batches_and_normalizes(self):
        client = OpenAICompatibleEmbeddingClient(
            "https://example.test/v1",
            "test-key",
            "test-embedding",
            dimension=2,
            dimensions=2,
            batch_size=2,
            max_retries=0,
        )
        payloads = []

        def fake_request(payload):
            payloads.append(payload)
            return {
                "data": [
                    {"index": index, "embedding": [float(index + 1), 1.0]}
                    for index, _ in enumerate(payload["input"])
                ]
            }

        client._request_sync = fake_request
        vectors = await client.embed(["a", "b", "c"])
        self.assertEqual(vectors.shape, (3, 2))
        self.assertEqual([len(payload["input"]) for payload in payloads], [2, 1])
        self.assertTrue(all(payload["dimensions"] == 2 for payload in payloads))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(3))

    def test_parse_pathrag_context_and_paths(self):
        context = """
-----high-level entity information-----
```csv
id,entity,type
0,A,DISEASE
```
-----high-level relationship information-----
```csv
id,source,target
0,A,B
```
-----Sources-----
```csv
id,content
0,"Evidence, with comma"
```
-----low-level entity information-----
```csv
id,entity,type
0,B,CELL
```
-----low-level relationship information-----
```csv
id,context
0,"The entity A is a DISEASE. The entity B is a CELL."
```
"""
        parsed = parse_pathrag_context(context)
        self.assertEqual(parsed["contexts"], ["Evidence, with comma"])
        self.assertEqual(parsed["paths"], [["A", "B"]])
        self.assertEqual(len(parsed["entities"]), 2)

    def test_parse_pathrag_context_strips_upstream_tabbed_headers(self):
        context = """
-----Sources-----
```csv
id,\tcontent
1,\t"Medical evidence"
```
"""
        self.assertEqual(
            parse_pathrag_context(context)["contexts"], ["Medical evidence"]
        )

    def test_official_silver_prefers_cheapest_correct(self):
        rows = {
            "vector": {
                "answer_correctness": 0.6,
                "input_tokens": 100,
                "output_tokens": 10,
                "total_time_ms": 20,
                "rouge_l": 0.4,
                "evidence_recall": 0.5,
            },
            "lightrag": {
                "answer_correctness": 0.62,
                "input_tokens": 120,
                "output_tokens": 10,
                "total_time_ms": 30,
                "rouge_l": 0.5,
                "evidence_recall": 0.6,
            },
            "pathrag": {
                "answer_correctness": 0.64,
                "input_tokens": 200,
                "output_tokens": 10,
                "total_time_ms": 40,
                "rouge_l": 0.6,
                "evidence_recall": 0.7,
            },
        }
        self.assertEqual(choose_official_silver(rows), ("vector", False))

    def test_official_silver_requires_near_best_correctness(self):
        rows = {
            "vector": {
                "answer_correctness": 0.60,
                "input_tokens": 100,
                "output_tokens": 10,
                "total_time_ms": 20,
                "rouge_l": 0.4,
                "evidence_recall": 0.5,
            },
            "lightrag": {
                "answer_correctness": 0.66,
                "input_tokens": 120,
                "output_tokens": 10,
                "total_time_ms": 30,
                "rouge_l": 0.5,
                "evidence_recall": 0.6,
            },
            "pathrag": {
                "answer_correctness": 0.70,
                "input_tokens": 200,
                "output_tokens": 10,
                "total_time_ms": 40,
                "rouge_l": 0.6,
                "evidence_recall": 0.7,
            },
        }
        self.assertEqual(choose_official_silver(rows), ("lightrag", False))

    def test_official_silver_falls_back_when_no_method_reaches_threshold(self):
        rows = {
            method: {
                "answer_correctness": score,
                "input_tokens": 100,
                "output_tokens": 10,
                "total_time_ms": 20,
                "rouge_l": 0.4,
                "evidence_recall": 0.5,
            }
            for method, score in {
                "vector": 0.40,
                "lightrag": 0.45,
                "pathrag": 0.50,
            }.items()
        }
        self.assertEqual(choose_official_silver(rows), ("pathrag", True))


if __name__ == "__main__":
    unittest.main()
