from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import sys
import time
import types
from pathlib import Path
from typing import Any

from .base import BackendResult, RAGBackend
from .model_client import (
    ANSWER_SYSTEM_PROMPT,
    TransformersChatClient,
    TransformersEmbeddingClient,
    generate_grounded_answer,
)


def _install_provider_import_shim() -> None:
    """Avoid PathRAG's eager imports of unused vLLM/Ollama provider stacks."""

    if "PathRAG.llm" in sys.modules:
        return
    shim = types.ModuleType("PathRAG.llm")

    async def unavailable(*_: Any, **__: Any) -> Any:
        raise RuntimeError("The PathRAG default provider is disabled; use the local adapter")

    shim.openai_complete = unavailable
    shim.openai_embedding = unavailable
    sys.modules["PathRAG.llm"] = shim


def _csv_records(text: str) -> list[dict[str, str]]:
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) < 2:
        return []
    headers = rows[0]
    return [
        {header: value for header, value in zip(headers, row)}
        for row in rows[1:]
        if any(value.strip() for value in row)
    ]


def parse_pathrag_context(context: str) -> dict[str, list[Any]]:
    labels = {
        "high_entities": "high-level entity information",
        "high_relations": "high-level relationship information",
        "sources": "Sources",
        "low_entities": "low-level entity information",
        "low_relations": "low-level relationship information",
    }
    sections: dict[str, list[dict[str, str]]] = {}
    for key, label in labels.items():
        match = re.search(
            rf"-----{re.escape(label)}-----\s*```csv\s*(.*?)\s*```",
            context,
            flags=re.DOTALL,
        )
        sections[key] = _csv_records(match.group(1)) if match else []
    path_rows = []
    for relation in sections["low_relations"]:
        relation_text = relation.get("context", "")
        nodes = re.findall(r"The entity (.+?) is a ", relation_text)
        if len(nodes) >= 2:
            path_rows.append(nodes)
    return {
        "contexts": [row["content"] for row in sections["sources"] if row.get("content")],
        "entities": sections["high_entities"] + sections["low_entities"],
        "relations": sections["high_relations"] + sections["low_relations"],
        "paths": path_rows,
    }


class PathRAGBackend(RAGBackend):
    """Adapter over unmodified BUPT-GAMMA PathRAG retrieval code."""

    method = "pathrag"

    def __init__(
        self,
        upstream_dir: str | Path,
        working_dir: str | Path,
        llm: TransformersChatClient,
        embedding: TransformersEmbeddingClient,
        chunk_tokens: int = 1200,
        chunk_overlap_tokens: int = 100,
        top_k: int = 40,
        max_context_tokens: int = 5000,
        generation_prompt: str = ANSWER_SYSTEM_PROMPT,
    ) -> None:
        self.upstream_dir = Path(upstream_dir).resolve()
        self.working_dir = Path(working_dir).resolve()
        self.llm = llm
        self.embedding = embedding
        self.chunk_tokens = int(chunk_tokens)
        self.chunk_overlap_tokens = int(chunk_overlap_tokens)
        self.top_k = int(top_k)
        self.max_context_tokens = int(max_context_tokens)
        self.generation_prompt = generation_prompt
        self.rag: Any = None
        self._query_param_cls: Any = None

    async def _initialize(self) -> None:
        if self.rag is not None:
            return
        self.working_dir.mkdir(parents=True, exist_ok=True)
        upstream_text = str(self.upstream_dir)
        if upstream_text not in sys.path:
            sys.path.insert(0, upstream_text)
        _install_provider_import_shim()
        from PathRAG import PathRAG, QueryParam
        from PathRAG.utils import EmbeddingFunc, logger

        if not logger.handlers:
            log_path = self.working_dir / "PathRAG.log"
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)

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
            func=embedding_callback,
            concurrent_limit=1,
        )
        self.rag = PathRAG(
            working_dir=str(self.working_dir),
            chunk_token_size=self.chunk_tokens,
            chunk_overlap_token_size=self.chunk_overlap_tokens,
            llm_model_func=llm_callback,
            llm_model_name=self.llm.model_name,
            llm_model_max_async=1,
            llm_model_max_token_size=32768,
            embedding_func=embedding_func,
            embedding_batch_num=self.embedding.batch_size,
            embedding_func_max_async=1,
            entity_extract_max_gleaning=1,
            enable_llm_cache=True,
            addon_params={"language": "English"},
        )
        self._query_param_cls = QueryParam

    async def index(self, corpus: str) -> dict[str, Any]:
        await self._initialize()
        marker = self.working_dir / "official_index_manifest.json"
        corpus_hash = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
        if marker.exists():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("corpus_sha256") == corpus_hash:
                return {**existing["index_stats"], "cached": True}
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
        }
        marker.write_text(
            json.dumps(
                {
                    "corpus_sha256": corpus_hash,
                    "upstream": "BUPT-GAMMA/PathRAG@32567bfc93605b8393996d5fa9ccdc0edbb865b2",
                    "path_pruning": {"alpha": 0.8, "threshold": 0.3, "max_hops": 3},
                    "index_stats": stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return stats

    async def query(self, question: str, question_id: str = "") -> dict[str, Any]:
        await self._initialize()
        total_started = time.perf_counter()
        before = self.llm.snapshot()
        retrieval_started = time.perf_counter()
        context = await self.rag.aquery(
            question,
            self._query_param_cls(
                mode="hybrid",
                only_need_context=True,
                top_k=self.top_k,
                max_token_for_text_unit=2000,
                max_token_for_global_context=1500,
                max_token_for_local_context=1500,
            ),
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        context = str(context or "")
        parsed = parse_pathrag_context(context)
        generation_started = time.perf_counter()
        generation = await generate_grounded_answer(
            self.llm,
            question,
            [context],
            system_prompt=self.generation_prompt,
            max_context_tokens=self.max_context_tokens,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        usage = self.llm.snapshot() - before
        result = BackendResult(
            question_id=question_id,
            method=self.method,
            answer=generation.text,
            contexts=parsed["contexts"],
            entities=parsed["entities"],
            relations=parsed["relations"],
            paths=parsed["paths"],
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            llm_calls=usage.calls,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            total_time_ms=(time.perf_counter() - total_started) * 1000,
        )
        return result.to_dict()
