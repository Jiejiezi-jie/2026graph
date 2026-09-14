from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import networkx as nx

from ..common.base import BackendResult, RAGBackend
from ..common.model_client import (
    ANSWER_SYSTEM_PROMPT,
    TransformersChatClient,
    TransformersEmbeddingClient,
    generate_grounded_answer,
)


class LightRAGBackend(RAGBackend):
    """Adapter over unmodified HKUDS LightRAG v1.5.7."""

    method = "lightrag"

    def __init__(
        self,
        working_dir: str | Path,
        llm: TransformersChatClient,
        embedding: TransformersEmbeddingClient,
        chunk_tokens: int = 1200,
        chunk_overlap_tokens: int = 100,
        top_k: int = 40,
        chunk_top_k: int = 5,
        max_context_tokens: int = 5000,
        generation_prompt: str = ANSWER_SYSTEM_PROMPT,
        max_entity_tokens: int = 1500,
        max_relation_tokens: int = 1500,
        max_total_tokens: int = 12000,
    ) -> None:
        self.working_dir = Path(working_dir)
        self.llm = llm
        self.embedding = embedding
        self.chunk_tokens = int(chunk_tokens)
        self.chunk_overlap_tokens = int(chunk_overlap_tokens)
        self.top_k = int(top_k)
        self.chunk_top_k = int(chunk_top_k)
        self.max_context_tokens = int(max_context_tokens)
        self.generation_prompt = generation_prompt
        # LightRAG's *internal* context budget for chunk payload selection. It is
        # decoupled from max_context_tokens (the budget we enforce when we render
        # the generation prompt): if the graph JSON shares the generation budget,
        # relation-heavy (Complex Reasoning) queries starve text chunks down to
        # 0-1 chunks and fall back to entity/relation JSON dumps as "context",
        # which the official evidence-recall judge cannot attribute to. These
        # caps bound the graph JSON so ranked text chunks get the budget instead.
        self.max_entity_tokens = int(max_entity_tokens)
        self.max_relation_tokens = int(max_relation_tokens)
        self.max_total_tokens = int(max_total_tokens)
        self.rag: Any = None

    def _index_identity(self) -> dict[str, Any]:
        return {
            "upstream": "HKUDS/LightRAG@28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c",
            "llm_model": self.llm.model_name,
            "embedding_model": getattr(
                self.embedding, "model_name", str(self.embedding.model_path)
            ),
            "embedding_dimension": self.embedding.dimension,
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
        }

    def _graph_stats(self) -> dict[str, Any]:
        graph_path = self.working_dir / "medical" / "graph_chunk_entity_relation.graphml"
        if not graph_path.exists() or graph_path.stat().st_size == 0:
            raise RuntimeError("LightRAG did not produce a non-empty GraphML file")
        graph = nx.read_graphml(graph_path)
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        if nodes == 0 or edges == 0:
            raise RuntimeError(
                f"LightRAG graph is incomplete: {nodes} nodes and {edges} edges"
            )
        return {
            "graph_path": str(graph_path),
            "graph_nodes": nodes,
            "graph_edges": edges,
        }

    def _prepare_index_identity(
        self, corpus_hash: str, identity: dict[str, Any]
    ) -> None:
        identity_path = self.working_dir / "official_index_identity.json"
        expected = {"corpus_sha256": corpus_hash, "index_identity": identity}
        if identity_path.exists():
            actual = json.loads(identity_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise RuntimeError(
                    "LightRAG index belongs to a different corpus or model configuration; "
                    "use a new results_dir"
                )
        else:
            identity_path.write_text(
                json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    async def _initialize(self) -> None:
        if self.rag is not None:
            return
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        self.working_dir.mkdir(parents=True, exist_ok=True)

        async def embedding_callback(texts: list[str], **kwargs: Any) -> Any:
            return await self.embedding.embed(texts, **kwargs)

        async def llm_callback(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            return await self.llm.complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=self.embedding.dimension,
            max_token_size=self.embedding.max_length,
            model_name=self.embedding.model_path.name,
            func=embedding_callback,
        )
        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            workspace="medical",
            llm_model_func=llm_callback,
            llm_model_name=self.llm.model_name,
            llm_model_max_async=1,
            embedding_func=embedding_func,
            embedding_batch_num=self.embedding.batch_size,
            embedding_func_max_async=1,
            chunk_token_size=self.chunk_tokens,
            chunk_overlap_token_size=self.chunk_overlap_tokens,
            max_parallel_insert=1,
            entity_extract_max_gleaning=1,
            entity_extraction_use_json=True,
            enable_llm_cache=True,
            enable_llm_cache_for_entity_extract=True,
        )
        await self.rag.initialize_storages()

    async def index(self, corpus: str) -> dict[str, Any]:
        await self._initialize()
        marker = self.working_dir / "official_index_manifest.json"
        corpus_hash = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
        identity = self._index_identity()
        if marker.exists():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("corpus_sha256") == corpus_hash and existing.get(
                "index_identity"
            ) == identity:
                return {
                    **existing["index_stats"],
                    **self._graph_stats(),
                    "cached": True,
                }
            raise RuntimeError(
                "Completed LightRAG index belongs to a different corpus or model "
                "configuration; use a new results_dir"
            )
        self._prepare_index_identity(corpus_hash, identity)
        before = self.llm.snapshot()
        started = time.perf_counter()
        await self.rag.ainsert(corpus)
        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = self.llm.snapshot() - before
        stats = {
            "method": self.method,
            "index_time_ms": elapsed_ms,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "llm_calls": usage.calls,
            "cached": False,
            **self._graph_stats(),
        }
        marker.write_text(
            json.dumps(
                {
                    "corpus_sha256": corpus_hash,
                    "upstream": "HKUDS/LightRAG@28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c",
                    "index_identity": identity,
                    "index_stats": stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return stats

    @staticmethod
    def _data_section(raw: dict[str, Any]) -> dict[str, list[Any]]:
        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        return {
            "entities": list(data.get("entities") or []),
            "relationships": list(data.get("relationships") or []),
            "chunks": list(data.get("chunks") or []),
        }

    async def query(self, question: str, question_id: str = "") -> dict[str, Any]:
        await self._initialize()
        from lightrag import QueryParam

        total_started = time.perf_counter()
        before = self.llm.snapshot()
        retrieval_started = time.perf_counter()
        raw = await self.rag.aquery_data(
            question,
            param=QueryParam(
                mode="hybrid",
                top_k=self.top_k,
                chunk_top_k=self.chunk_top_k,
                max_entity_tokens=self.max_entity_tokens,
                max_relation_tokens=self.max_relation_tokens,
                max_total_tokens=self.max_total_tokens,
                enable_rerank=False,
            ),
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        section = self._data_section(raw)
        contexts = [
            str(chunk.get("content", ""))
            for chunk in section["chunks"]
            if isinstance(chunk, dict) and chunk.get("content")
        ]
        if not contexts:
            # LightRAG can answer from graph evidence even when no source chunk
            # survives its final chunk selection. This is a degraded-retrieval
            # signal, not ordinary evidence: the official evidence-recall judge
            # cannot attribute text evidence to entity/relation JSON, so rows
            # that land here will (correctly) score low on evidence_recall.
            fallback_contexts = [
                "Entities:\n"
                + json.dumps(section["entities"], ensure_ascii=False)
                + "\nRelationships:\n"
                + json.dumps(section["relationships"], ensure_ascii=False)
            ]
        else:
            fallback_contexts = []
        generation_started = time.perf_counter()
        generation = await generate_grounded_answer(
            self.llm,
            question,
            contexts if contexts else fallback_contexts,
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
            entities=section["entities"],
            relations=section["relationships"],
            paths=[],
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            llm_calls=usage.calls,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            total_time_ms=(time.perf_counter() - total_started) * 1000,
        )
        row = result.to_dict()
        row["retrieval_metadata"] = raw.get("metadata", {}) if isinstance(raw, dict) else {}
        return row

    async def close(self) -> None:
        if self.rag is not None:
            await self.rag.finalize_storages()
            self.rag = None
