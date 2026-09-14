#!/usr/bin/env python
"""从已有 PathRAG graphml 恢复缺失的索引(不重新调用 LLM 抽取)。

背景: result/api/indexes/pathrag/ 的 graphml(4568 节点/9432 边)是完整的实体抽取产物,
但 vdb_entities/vdb_relationships 为空、kv_store_text_chunks/full_docs 为空、无 manifest,
导致 pathrag_backend.index() 会触发全量 ainsert 重新抽取。本脚本从 graphml + 语料
重建缺失存储,使索引可被 query 使用且 index() 走 cached 分支。

只重建存储文件,不修改 graphml,不调用任何 LLM。
用法:
  PYTHONPATH=. .venv_official/bin/python src/backend/pathRAG/restore_pathrag_index.py
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path

import networkx as nx

PROJECT = Path(__file__).resolve().parents[3]
INDEX_DIR = PROJECT / "result/api" / "indexes" / "pathrag"
BENCH_DIR = PROJECT / "data" / "vendor" / "GraphRAG-Benchmark"
GRAPH = INDEX_DIR / "graph_chunk_entity_relation.graphml"

CHUNK_TOKENS = 1200
CHUNK_OVERLAP = 100
TIKTOKEN_MODEL = "gpt-4o-mini"
EMB_DIM = 1024
UPSTREAM = "BUPT-GAMMA/PathRAG@32567bfc93605b8393996d5fa9ccdc0edbb865b2"


def mdhash(content: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(content.encode()).hexdigest()


def unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def split_multi(s: str) -> list[str]:
    """graphml 属性形如 "seg1"<SEP>"seg2"...;拆段并去每段外层引号。"""
    parts = [unquote(p) for p in str(s or "").split("<SEP>")]
    return [p for p in parts if p]


def load_benchmark() -> tuple[str, Path]:
    corpus_path = BENCH_DIR / "Datasets" / "Corpus" / "medical.json"
    return json.loads(corpus_path.read_text(encoding="utf-8"))["context"], corpus_path


# ---------------- chunking (与上游 operate.chunking_by_token_size 一致) ----------------
def chunk_by_tokens(content: str):
    import tiktoken

    enc = tiktoken.encoding_for_model(TIKTOKEN_MODEL)
    tokens = enc.encode(content)
    results = []
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    for index, start in enumerate(range(0, len(tokens), step)):
        chunk_content = enc.decode(tokens[start : start + CHUNK_TOKENS])
        results.append(
            {
                "tokens": min(CHUNK_TOKENS, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


def build_text_chunks(corpus: str) -> tuple[dict, dict, dict]:
    """重建 kv_store_text_chunks.json + kv_store_full_docs.json + 各 chunk 文本。"""
    doc_id = mdhash(corpus.strip(), prefix="doc-")
    full_docs = {doc_id: {"content": corpus.strip()}}
    chunks = chunk_by_tokens(corpus)
    text_store = {}
    chunk_texts = {}
    for c in chunks:
        cid = mdhash(c["content"], prefix="chunk-")
        chunk_texts[cid] = c["content"]
        text_store[cid] = {
            "chunk_order_index": c["chunk_order_index"],
            "content": c["content"],
            "full_doc_id": doc_id,
            "tokens": c["tokens"],
        }
    return full_docs, text_store, chunk_texts


def graphml_entities_relations():
    G = nx.read_graphml(GRAPH)
    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))
    entities = []
    for nid, attrs in nodes:
        entities.append(
            {
                "id": mdhash(nid, prefix="ent-"),
                "name": nid,  # 保留引号,与 graphml/smoke vdb 一致
                "desc": "".join(split_multi(attrs.get("description", ""))),
            }
        )
    relations = []
    for u, v, attrs in edges:
        rel = {
            "id": mdhash(u + v, prefix="rel-"),
            "src_id": u,
            "tgt_id": v,
            "desc": "".join(split_multi(attrs.get("description", ""))),
            "keywords": "".join(split_multi(attrs.get("keywords", ""))),
        }
        relations.append(rel)
    return entities, relations


def vecs_to_nano_db(meta_list: list[dict], vecs: list[list[float]], emb_dim: int) -> dict:
    """构造 nano-vectordb 落盘格式: data(无 __vector__) + matrix(base64 float32)。"""
    import numpy as np

    assert len(meta_list) == len(vecs), f"{len(meta_list)} != {len(vecs)}"
    mat = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    mat = mat / norms
    data = []
    for m, row in zip(meta_list, mat):
        d = {k: v for k, v in m.items() if k != "desc" and k != "keywords"}
        data.append(d)
    return {
        "embedding_dim": emb_dim,
        "data": data,
        "matrix": base64.b64encode(mat.tobytes()).decode(),
    }


async def embed_texts(client, texts: list[str], batch: int) -> list[list[float]]:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        result = await client.embed(texts[i : i + batch])
        # result: np.ndarray shape [n, emb_dim]
        for row in result:
            vecs.append(row.tolist() if hasattr(row, "tolist") else list(row))
        print(f"  embed {min(i+batch, len(texts))}/{len(texts)}", flush=True)
    return vecs


def main() -> None:
    import sys

    sys.path.insert(0, str(PROJECT))
    from src.backend.common.model_client import build_embedding_client

    print("== 1/4 语料切块(重建 text_chunks/full_docs) ==", flush=True)
    corpus, _ = load_benchmark()
    full_docs, text_store, chunk_texts = build_text_chunks(corpus)
    print(f"  doc: 1, chunks: {len(text_store)}", flush=True)

    # 校验: 重建的 chunk id 必须与 vdb_chunks.json 现有 199 个 id 一致
    bak_vdb = json.loads((INDEX_DIR / "vdb_chunks.json").read_text(encoding="utf-8"))
    existing_ids = {d["__id__"] for d in bak_vdb["data"]}
    new_ids = set(text_store.keys())
    assert new_ids == existing_ids, (
        f"chunk id 不一致: 新 {len(new_ids)} vs 现有 {len(existing_ids)}; "
        f"独有新={len(new_ids-existing_ids)} 独有旧={len(existing_ids-new_ids)}"
    )
    print("  chunk id 与现有 vdb_chunks 一致 ✓", flush=True)

    print("== 2/4 读 graphml 实体/关系 ==", flush=True)
    entities, relations = graphml_entities_relations()
    print(f"  实体 {len(entities)}, 关系 {len(relations)}", flush=True)
    # 图统计写入 manifest 用
    G = nx.read_graphml(GRAPH)

    print("== 3/4 生成实体/关系向量 ==", flush=True)
    cfg = {
        "backend": "transformers",
        "model_path": str(PROJECT / "models" / "bge-m3"),
        "device": "cuda:0",
        "batch_size": 16,
        "max_length": 2048,
        "normalize": True,
    }
    emb = build_embedding_client(cfg)
    try:
        ent_texts = [e["name"] + e["desc"] for e in entities]
        print(f"  实体向量 {len(ent_texts)} 条...", flush=True)
        ent_vecs = asyncio.run(embed_texts(emb, ent_texts, cfg["batch_size"]))
        rel_texts = [r["keywords"] + r["src_id"] + r["tgt_id"] + r["desc"] for r in relations]
        print(f"  关系向量 {len(rel_texts)} 条...", flush=True)
        rel_vecs = asyncio.run(embed_texts(emb, rel_texts, cfg["batch_size"]))
    finally:
        emb.unload()

    print("== 4/4 写存储文件 ==", flush=True)
    ent_meta = [{"__id__": e["id"], "entity_name": e["name"]} for e in entities]
    rel_meta = [{"__id__": r["id"], "src_id": r["src_id"], "tgt_id": r["tgt_id"]} for r in relations]
    (INDEX_DIR / "vdb_entities.json").write_text(
        json.dumps(vecs_to_nano_db(ent_meta, ent_vecs, EMB_DIM)), encoding="utf-8"
    )
    (INDEX_DIR / "vdb_relationships.json").write_text(
        json.dumps(vecs_to_nano_db(rel_meta, rel_vecs, EMB_DIM)), encoding="utf-8"
    )
    (INDEX_DIR / "kv_store_text_chunks.json").write_text(
        json.dumps(text_store), encoding="utf-8"
    )
    (INDEX_DIR / "kv_store_full_docs.json").write_text(
        json.dumps(full_docs), encoding="utf-8"
    )
    print("  vdb_entities/vdb_relationships/kv_store_text_chunks/kv_store_full_docs 已写", flush=True)

    # manifest: corpus hash + identity(与 identity 文件一致) + index_stats
    corpus_hash = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    identity_file = json.loads((INDEX_DIR / "official_index_identity.json").read_text(encoding="utf-8"))
    identity = identity_file["index_identity"]
    stats = {
        "method": "pathrag",
        "index_time_ms": 0,  # 恢复产物,非真实索引耗时
        "input_tokens": 0,
        "output_tokens": 0,
        "llm_calls": 0,
        "cached": False,
        "graph_path": str(INDEX_DIR / "graph_chunk_entity_relation.graphml"),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }
    manifest = {
        "corpus_sha256": corpus_hash,
        "upstream": UPSTREAM,
        "index_identity": identity,
        "path_pruning": {"alpha": 0.8, "threshold": 0.3, "max_hops": 3},
        "index_stats": stats,
    }
    (INDEX_DIR / "official_index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("  official_index_manifest.json 已写", flush=True)
    print("\n恢复完成 ✓ 索引目录:", INDEX_DIR, flush=True)


if __name__ == "__main__":
    main()
