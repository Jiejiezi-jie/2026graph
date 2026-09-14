#!/usr/bin/env python
"""PathRAG 检索调优对照：cosine_better_than_threshold + top_k 对耗时与证据聚焦的影响。

跑 1 道题、多组参数对照,只检索不重复建索引、不改源码/索引。
用法(用户终端,需 GPU+key):
  export PYTHONPATH=.
  export DEEPSEEK_API_KEY=...
  .venv_official/bin/python src/backend/pathRAG/pathrag_tune_probe.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

QUESTION = (
    "Why are fair skin and organ transplantation both risk factors "
    "for basal cell carcinoma?"
)


def _parse(rag, raw) -> dict:
    from src.backend.pathRAG.pathrag_backend import parse_pathrag_context

    parsed = parse_pathrag_context(str(raw or ""))
    return parsed


async def probe(top_k: int, threshold: float, question: str) -> dict:
    config = json.loads(
        (PROJECT / "configs" / "official_api_gpu.json").read_text(encoding="utf-8")
    )
    # 本机无 GPU 调试: 允许 DSH_PROBE_CPU=1 走 CPU embedding(仅实验,不影响正式 config)
    if __import__("os").environ.get("DSH_PROBE_CPU"):
        config["embedding"]["device"] = "cpu"
    from src.backend.common.model_client import (
        build_chat_client,
        build_embedding_client,
    )
    from src.backend.pathRAG.pathrag_backend import PathRAGBackend

    llm = build_chat_client(config["llm"])
    emb = build_embedding_client(config["embedding"])
    backend = PathRAGBackend(
        upstream_dir=PROJECT / "deps" / "PathRAG",
        working_dir=PROJECT / "result/api" / "indexes" / "pathrag",
        top_k=top_k,
        llm=llm,
        embedding=emb,
        chunk_tokens=config["chunk_tokens"],
        chunk_overlap_tokens=config["chunk_overlap_tokens"],
        max_context_tokens=config["max_context_tokens"],
        generation_prompt=config["generation_prompt"],
    )
    try:
        await backend._initialize()
        rag = backend.rag
        # 注入检索阈值(直接改 storage 实例字段)
        rag.entities_vdb.cosine_better_than_threshold = threshold
        rag.relationships_vdb.cosine_better_than_threshold = threshold
        # backend._initialize 已设 _query_param_cls
        param_cls = backend._query_param_cls

        t0 = time.perf_counter()
        raw = await rag.aquery(
            question,
            param_cls(
                mode="hybrid",
                only_need_context=True,
                top_k=top_k,
                max_token_for_text_unit=2000,
                max_token_for_global_context=1500,
                max_token_for_local_context=1500,
            ),
        )
        elapsed = time.perf_counter() - t0
        parsed = _parse(rag, raw)
        contexts = parsed.get("contexts") or []
        ctx_text = " ".join(str(c) for c in contexts)
        # 证据聚焦度: 标准证据句(取该题 ground truth 关键短语)是否出现在 contexts
        ev_key = "immunosuppression"
        return {
            "top_k": top_k,
            "threshold": threshold,
            "seconds": round(elapsed, 1),
            "n_contexts": len(contexts),
            "context_chars": len(ctx_text),
            "has_immunosuppression": ("immunosuppression" in ctx_text.lower()),
            "n_entities": len(parsed.get("entities") or []),
        }
    finally:
        await backend.close()
        llm.unload()
        emb.unload()


def main() -> None:
    question = QUESTION
    runs = [(40, 0.2), (40, 0.45), (20, 0.45), (15, 0.5)]
    print(f"问题: {question}\n", flush=True)
    for top_k, thr in runs:
        print(f"--- top_k={top_k} threshold={thr} ---", flush=True)
        try:
            res = asyncio.run(probe(top_k, thr, question))
            print(json.dumps(res, ensure_ascii=False), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
