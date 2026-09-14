from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.backend.lightRAG.lightrag_backend import LightRAGBackend
from src.backend.common.model_client import Generation, UsageSnapshot
from src.backend.pathRAG.pathrag_backend import PathRAGBackend


ROOT = Path(__file__).parents[1]
UPSTREAMS_AVAILABLE = (ROOT / "deps" / "LightRAG").exists() and (
    ROOT / "deps" / "PathRAG"
).exists()

CORPUS = (
    "Basal cell carcinoma is a common skin cancer. Ultraviolet radiation damages "
    "DNA in skin cells and increases basal cell carcinoma risk. Organ transplant "
    "recipients use immunosuppressive medicines. Immunosuppression weakens immune "
    "surveillance and also increases basal cell carcinoma risk."
)
QUESTION = "Why do ultraviolet radiation and immunosuppression increase BCC risk?"


class DeterministicEmbedding:
    model_path = Path("deterministic-embedding")
    model_name = "deterministic-embedding"
    max_length = 8191
    dimension = 8
    batch_size = 8

    async def embed(self, texts, **kwargs):
        vector = np.ones(self.dimension, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        return np.stack([vector.copy() for _ in texts])

    def unload(self):
        return None


class DeterministicLLM:
    model_name = "deterministic-llm"
    temperature = 0.0

    def __init__(self):
        self.usage = UsageSnapshot()

    def _record(self, text: str) -> Generation:
        self.usage = UsageSnapshot(
            self.usage.input_tokens + 20,
            self.usage.output_tokens + 10,
            self.usage.calls + 1,
        )
        return Generation(text, 20, 10)

    async def generate(self, prompt, system_prompt=None, **kwargs):
        return self._record(
            "Both exposures impair protection against DNA-damaged skin cells."
        )

    async def complete(
        self, prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        if kwargs.get("keyword_extraction"):
            return self._record(
                '{"high_level_keywords":["skin cancer risk"],'
                '"low_level_keywords":["ultraviolet radiation",'
                '"immunosuppression","basal cell carcinoma"]}'
            ).text
        if kwargs.get("response_format"):
            if "high_level_keywords" in prompt:
                return self._record(
                    '{"high_level_keywords":["skin cancer risk"],'
                    '"low_level_keywords":["ultraviolet radiation",'
                    '"immunosuppression","basal cell carcinoma"]}'
                ).text
            if history_messages:
                return self._record('{"entities":[],"relationships":[]}').text
            return self._record(
                '{"entities":['
                '{"name":"Basal Cell Carcinoma","type":"Disease",'
                '"description":"A common skin cancer."},'
                '{"name":"Ultraviolet Radiation","type":"Risk Factor",'
                '"description":"Damages DNA in skin cells."},'
                '{"name":"Immunosuppression","type":"Risk Factor",'
                '"description":"Weakens immune surveillance."}],'
                '"relationships":['
                '{"source":"Ultraviolet Radiation",'
                '"target":"Basal Cell Carcinoma","keywords":"risk",'
                '"description":"Ultraviolet radiation increases BCC risk."},'
                '{"source":"Immunosuppression",'
                '"target":"Basal Cell Carcinoma","keywords":"risk",'
                '"description":"Immunosuppression increases BCC risk."}]}'
            ).text
        if "MANY entities were missed" in prompt:
            return self._record("<|COMPLETE|>").text
        if "Entity_types:" in prompt:
            return self._record(
                '("entity"<|>"Basal Cell Carcinoma"<|>"category"<|>'
                '"A common skin cancer.")##'
                '("entity"<|>"Ultraviolet Radiation"<|>"category"<|>'
                '"A risk factor that damages DNA.")##'
                '("entity"<|>"Immunosuppression"<|>"category"<|>'
                '"A risk factor that weakens immune surveillance.")##'
                '("relationship"<|>"Ultraviolet Radiation"<|>'
                '"Basal Cell Carcinoma"<|>"Increases BCC risk."<|>'
                '"risk"<|>9)##'
                '("relationship"<|>"Immunosuppression"<|>'
                '"Basal Cell Carcinoma"<|>"Increases BCC risk."<|>'
                '"risk"<|>9)<|COMPLETE|>'
            ).text
        if "high_level_keywords" in prompt:
            return self._record(
                '{"high_level_keywords":["skin cancer risk"],'
                '"low_level_keywords":["ultraviolet radiation",'
                '"immunosuppression","basal cell carcinoma"]}'
            ).text
        return self._record("No additional information.").text

    def snapshot(self):
        return self.usage

    def unload(self):
        return None


@unittest.skipUnless(UPSTREAMS_AVAILABLE, "pinned upstream sources are not installed")
class PinnedUpstreamIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_lightrag_indexes_graph_and_retrieves(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LightRAGBackend(
                directory,
                DeterministicLLM(),
                DeterministicEmbedding(),
                chunk_tokens=300,
                chunk_overlap_tokens=30,
                top_k=10,
                chunk_top_k=3,
                max_context_tokens=5000,
            )
            try:
                stats = await backend.index(CORPUS)
                row = await backend.query(QUESTION, "smoke-light")
            finally:
                await backend.close()
            self.assertGreater(stats["graph_nodes"], 0)
            self.assertGreater(stats["graph_edges"], 0)
            self.assertTrue(row["contexts"])
            self.assertTrue(row["entities"])
            self.assertTrue(row["relations"])

    async def test_pathrag_indexes_graph_and_retrieves(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = PathRAGBackend(
                ROOT / "deps" / "PathRAG",
                directory,
                DeterministicLLM(),
                DeterministicEmbedding(),
                chunk_tokens=300,
                chunk_overlap_tokens=30,
                top_k=10,
                max_context_tokens=5000,
            )
            try:
                await backend._initialize()
                with patch("PathRAG.operate.time.sleep", return_value=None):
                    stats = await backend.index(CORPUS)
                row = await backend.query(QUESTION, "smoke-path")
            finally:
                await backend.close()
            self.assertGreater(stats["graph_nodes"], 0)
            self.assertGreater(stats["graph_edges"], 0)
            self.assertTrue(row["contexts"])
            self.assertTrue(row["entities"])
            self.assertTrue(row["relations"])


if __name__ == "__main__":
    unittest.main()
