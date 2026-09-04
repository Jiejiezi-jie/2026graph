import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from src.official_analysis import choose_official_silver
from src.official_backends.lightrag_backend import LightRAGBackend
from src.official_backends.model_client import Generation, UsageSnapshot
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
                "answer_correctness": 0.8,
                "input_tokens": 120,
                "output_tokens": 10,
                "total_time_ms": 30,
                "rouge_l": 0.5,
                "evidence_recall": 0.6,
            },
            "pathrag": {
                "answer_correctness": 0.9,
                "input_tokens": 200,
                "output_tokens": 10,
                "total_time_ms": 40,
                "rouge_l": 0.6,
                "evidence_recall": 0.7,
            },
        }
        self.assertEqual(choose_official_silver(rows), ("vector", False))


if __name__ == "__main__":
    unittest.main()
