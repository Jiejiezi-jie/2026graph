from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

from ..common.base import BackendResult, RAGBackend
from ..common.model_client import (
    ANSWER_SYSTEM_PROMPT,
    TransformersChatClient,
    TransformersEmbeddingClient,
    generate_grounded_answer,
)


def chunk_by_token_window(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    token_ids = encoding.encode(text)
    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(token_ids), step):
        chunk = encoding.decode(token_ids[start : start + size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(token_ids):
            break
    return chunks


class VectorRAGBackend(RAGBackend):
    method = "vector"

    def __init__(
        self,
        working_dir: str | Path,
        llm: TransformersChatClient,
        embedding: TransformersEmbeddingClient,
        chunk_tokens: int = 1200,
        chunk_overlap_tokens: int = 100,
        top_k: int = 5,
        max_context_tokens: int = 5000,
        generation_prompt: str = ANSWER_SYSTEM_PROMPT,
    ) -> None:
        self.working_dir = Path(working_dir)
        self.llm = llm
        self.embedding = embedding
        self.chunk_tokens = int(chunk_tokens)
        self.chunk_overlap_tokens = int(chunk_overlap_tokens)
        self.top_k = int(top_k)
        self.max_context_tokens = int(max_context_tokens)
        self.generation_prompt = generation_prompt
        self.chunks: list[str] = []
        self.vectors: np.ndarray | None = None

    def _metadata_path(self) -> Path:
        return self.working_dir / "index.json"

    def _vectors_path(self) -> Path:
        return self.working_dir / "vectors.npy"

    def _index_identity(self) -> dict[str, Any]:
        return {
            "embedding_model": getattr(
                self.embedding, "model_name", str(self.embedding.model_path)
            ),
            "embedding_dimension": self.embedding.dimension,
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
        }

    def _load_cached(
        self, corpus_hash: str, identity: dict[str, Any]
    ) -> dict[str, Any] | None:
        metadata_path = self._metadata_path()
        vectors_path = self._vectors_path()
        if not metadata_path.exists() or not vectors_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stored_identity = metadata.get("index_identity")
        legacy_matches = (
            stored_identity is None
            and metadata.get("embedding_model") == identity["embedding_model"]
            and metadata.get("chunk_tokens") == self.chunk_tokens
            and metadata.get("chunk_overlap_tokens") == self.chunk_overlap_tokens
        )
        if metadata.get("corpus_sha256") != corpus_hash or not (
            stored_identity == identity or legacy_matches
        ):
            return None
        self.chunks = list(metadata["chunks"])
        self.vectors = np.load(vectors_path)
        if self.vectors.ndim != 2 or self.vectors.shape[1] != self.embedding.dimension:
            raise RuntimeError("Cached vector index has the wrong embedding dimension")
        return {**metadata["index_stats"], "cached": True}

    def load_cached_index(self, corpus: str) -> dict[str, Any]:
        """Load a compatible index without permitting an implicit rebuild."""

        cached = self._load_cached(
            hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
            self._index_identity(),
        )
        if cached is None:
            raise RuntimeError(
                "--skip-index requested, but no compatible cached vector index exists"
            )
        return cached

    async def index(self, corpus: str) -> dict[str, Any]:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        corpus_hash = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
        metadata_path = self._metadata_path()
        vectors_path = self._vectors_path()
        identity = self._index_identity()
        cached = self._load_cached(corpus_hash, identity)
        if cached is not None:
            return cached

        started = time.perf_counter()
        self.chunks = chunk_by_token_window(
            corpus, self.chunk_tokens, self.chunk_overlap_tokens
        )
        self.vectors = await self.embedding.embed(self.chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000
        np.save(vectors_path, self.vectors)
        stats = {
            "method": self.method,
            "chunks": len(self.chunks),
            "index_time_ms": elapsed_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "llm_calls": 0,
            "cached": False,
        }
        metadata_path.write_text(
            json.dumps(
                {
                    "corpus_sha256": corpus_hash,
                    "embedding_model": identity["embedding_model"],
                    "index_identity": identity,
                    "chunk_tokens": self.chunk_tokens,
                    "chunk_overlap_tokens": self.chunk_overlap_tokens,
                    "chunks": self.chunks,
                    "index_stats": stats,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return stats

    async def query(self, question: str, question_id: str = "") -> dict[str, Any]:
        if self.vectors is None or not self.chunks:
            raise RuntimeError("Vector index is not initialized")
        total_started = time.perf_counter()
        before = self.llm.snapshot()
        retrieval_started = time.perf_counter()
        query_vector = (await self.embedding.embed([question]))[0]
        scores = self.vectors @ query_vector
        count = min(self.top_k, len(scores))
        candidates = np.argpartition(-scores, count - 1)[:count]
        selected = candidates[np.argsort(-scores[candidates])]
        contexts = [self.chunks[int(index)] for index in selected]
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        generation_started = time.perf_counter()
        generation = await generate_grounded_answer(
            self.llm,
            question,
            contexts,
            system_prompt=self.generation_prompt,
            max_context_tokens=self.max_context_tokens,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        usage = self.llm.snapshot() - before
        result = BackendResult(
            question_id=question_id,
            method=self.method,
            answer=generation.text,
            contexts=contexts,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            llm_calls=usage.calls,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            total_time_ms=(time.perf_counter() - total_started) * 1000,
        )
        row = result.to_dict()
        row["retrieved_chunk_ids"] = [int(index) for index in selected]
        row["retrieval_scores"] = [float(scores[int(index)]) for index in selected]
        return row
