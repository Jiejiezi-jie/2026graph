"""Offline probe: re-run LightRAG retrieval for a few P0 questions after the
budget fix and inspect the payload chunk selection. Relies on LightRAG's LLM
response cache for keyword extraction; if the cache misses, the stub LLM raises
and we know the probe cannot run offline.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lightrag import QueryParam  # noqa: E402

from src.official_backends.lightrag_backend import LightRAGBackend  # noqa: E402


class _OfflineLLM:
    """Raises if any LLM call escapes the LightRAG cache."""

    model_name = "deepseek-chat"

    async def complete(self, *args, **kwargs):
        raise RuntimeError(
            "LLM call escaped the cache (network needed); cannot probe offline"
        )

    def snapshot(self):
        return type("S", (), {"input_tokens": 0, "output_tokens": 0, "calls": 0})()

    def unload(self):
        return None


async def main() -> None:
    config = json.loads((PROJECT / "configs/official_api.json").read_text(encoding="utf-8"))
    questions = {
        r["question_id"]: r
        for r in (
            json.loads(line)
            for line in (PROJECT / "results_api/p0/vector_evaluated.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    targets = ["Medical-c09d2bcb", "Medical-2a5688d0", "Medical-7fba506d",
               "Medical-b7f57756", "Medical-0535a6b1"]
    payloads: dict[str, list[str]] = {}

    from src.official_backends.model_client import build_embedding_client

    embedding = build_embedding_client(config["embedding"])
    backend = LightRAGBackend(
        working_dir=PROJECT / "results_api/indexes/lightrag",
        llm=_OfflineLLM(),
        embedding=embedding,
        chunk_tokens=config["chunk_tokens"],
        chunk_overlap_tokens=config["chunk_overlap_tokens"],
        top_k=config["lightrag_top_k"],
        chunk_top_k=config["lightrag_chunk_top_k"],
        max_context_tokens=config["max_context_tokens"],
        max_entity_tokens=config["lightrag_max_entity_tokens"],
        max_relation_tokens=config["lightrag_max_relation_tokens"],
        max_total_tokens=config["lightrag_max_total_tokens"],
    )
    await backend._initialize()
    try:
        for qid in targets:
            q = questions[qid]["question"]
            raw = await backend.rag.aquery_data(
                q,
                param=QueryParam(
                    mode="hybrid",
                    top_k=backend.top_k,
                    chunk_top_k=backend.chunk_top_k,
                    max_entity_tokens=backend.max_entity_tokens,
                    max_relation_tokens=backend.max_relation_tokens,
                    max_total_tokens=backend.max_total_tokens,
                    enable_rerank=False,
                ),
            )
            data = raw.get("data", {})
            chunks = [c for c in data.get("chunks", []) if isinstance(c, dict)]
            info = raw.get("metadata", {}).get("processing_info", {})
            sizes = [len(c.get("content", "")) for c in chunks]
            head = (chunks[0].get("content", "")[:80].replace("\n", " ")
                    if chunks else "")
            print(f"{qid}: final_chunks={len(chunks)} sizes={sizes}")
            print(f"   meta merged={info.get('merged_chunks_count')} final={info.get('final_chunks_count')} "
                  f"ents={info.get('entities_after_truncation')} rels={info.get('relations_after_truncation')}")
            print(f"   head: {head!r}")
            payloads[qid] = [c.get("content", "") for c in chunks]
            # Evidence sanity check: does any delivered chunk text carry the
            # ground-truth evidence wording? (lexical proxy, paraphrases miss)
            gt = questions[qid].get("ground_truth", "")
            joined = " ".join(payloads[qid]).lower()
            probe_hit = any(w in joined for w in
                            [w for w in gt.lower().replace(".", " ").split()
                             if len(w) > 6][:6])
            print(f"   gt-keyword hit in payload: {probe_hit}")
        (PROJECT / "results_api/p0/_probe_lightrag_new_chunks.json").write_text(
            json.dumps(payloads, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    finally:
        embedding.unload()
        backend.rag = None


if __name__ == "__main__":
    asyncio.run(main())
