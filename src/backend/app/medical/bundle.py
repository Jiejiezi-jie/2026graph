import hashlib
import json
from pathlib import Path

import numpy as np

from app.domain.errors import AppError
from app.domain.models import CorpusDocument


class MedicalBundle:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.manifest = self.read("bundle.json")
        if self.manifest.get("schema_version") != 1:
            raise AppError("INVALID_MEDICAL_BUNDLE", "Medical bundle 版本不受支持。")
        self.graph_dirs = {
            "lightrag": self.root / "indexes/lightrag/medical",
            "pathrag": self.root / "indexes/lightrag/medical",
        }
        hashes = set()
        stored_hashes = set()
        counts = {}
        for method, directory in {"lightrag": self.graph_dirs["lightrag"]}.items():
            for name in ("official_index_identity.json", "official_index_manifest.json"):
                metadata = self.read(f"indexes/{method}/{name}")
                hashes.add(metadata.get("corpus_sha256"))
                self.check_identity(metadata.get("index_identity", {}))
            self.require(directory / "graph_chunk_entity_relation.graphml")
            chunks = self.read(directory / "kv_store_text_chunks.json")
            docs = self.read(directory / "kv_store_full_docs.json")
            if not chunks or not docs:
                raise AppError("INVALID_MEDICAL_BUNDLE", "Medical 原文或 Chunk 为空。")
            counts[method + "_chunks"] = len(chunks)
            # This imported release indexes one concatenated Medical corpus.
            if len(docs) != 1 or "content" not in next(iter(docs.values())):
                raise AppError("INVALID_MEDICAL_BUNDLE", "Medical 原文存储结构与此接入版本不匹配。")
            content = next(iter(docs.values()))["content"]
            stored_hashes.add(hashlib.sha256(content.encode("utf-8")).hexdigest())
            for name in ("chunks", "entities", "relationships"):
                vdb = self.read(directory / f"vdb_{name}.json")
                if vdb.get("embedding_dim") != 1024:
                    raise AppError("INVALID_MEDICAL_BUNDLE", "Medical 向量维度必须为 1024 dimension。")
                if not vdb.get("data"):
                    raise AppError("INVALID_MEDICAL_BUNDLE", f"Medical {method}/{name} 向量索引为空。")
        counts["pathrag_chunks"] = counts["lightrag_chunks"]
        vector = self.read("indexes/vector/index.json")
        self.check_identity(vector.get("index_identity", {}))
        hashes.add(vector.get("corpus_sha256"))
        if len(hashes) != 1 or None in hashes:
            raise AppError("INVALID_MEDICAL_BUNDLE", "Medical 各索引的原文 corpus 哈希不一致。")
        if len(stored_hashes) != 1:
            raise AppError("INVALID_MEDICAL_BUNDLE", "LightRAG 与 PathRAG 存储的全文原文不一致。")
        self.corpus_hash = hashes.pop()
        self.stored_corpus_hash = stored_hashes.pop()
        self.require(self.root / "indexes/vector/vectors.npy")
        try:
            vectors = np.load(self.root / "indexes/vector/vectors.npy", mmap_mode="r", allow_pickle=False)
            if vectors.ndim != 2 or vectors.shape[1] != 1024:
                raise AppError("INVALID_MEDICAL_BUNDLE", "Vector 向量维度 dimension 应为 1024。")
            if vectors.shape[0] != len(vector.get("chunks", [])) or not vectors.shape[0]:
                raise AppError("INVALID_MEDICAL_BUNDLE", "Vector 向量数量与 Chunk 数量不匹配。")
            if not np.isfinite(vectors).all():
                raise AppError("INVALID_MEDICAL_BUNDLE", "Vector 向量包含无效数值。")
            counts["vector_chunks"] = vectors.shape[0]
        except (ValueError, OSError) as exc:
            raise AppError("INVALID_MEDICAL_BUNDLE", "Vector 向量文件无法读取。") from exc
        self.require(self.root / self.manifest.get("router", {}).get("artifact", "router.joblib"))
        self.document = CorpusDocument(corpus_name=self.manifest.get("corpus_name", "Medical"),
                                       subset="medical", context=content,
                                       character_count=len(content), word_count=len(content.split()))
        self.counts = counts

    @staticmethod
    def check_identity(identity):
        expected = {"embedding_model": "bge-m3", "embedding_dimension": 1024,
                    "chunk_tokens": 1200, "chunk_overlap_tokens": 100}
        if any(identity.get(key) != value for key, value in expected.items()):
            raise AppError("INVALID_MEDICAL_BUNDLE", "Medical 索引的模型、维度或分块参数不匹配。")

    def require(self, path):
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise AppError("INVALID_MEDICAL_BUNDLE", f"Medical bundle 缺少文件：{path.name}")

    def read(self, relative):
        path = self.root / relative
        self.require(path)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise AppError("INVALID_MEDICAL_BUNDLE", f"Medical 文件无法读取：{path.name}") from exc

    def summary(self):
        return {"declared_corpus_sha256": self.corpus_hash,
                "stored_corpus_sha256": self.stored_corpus_hash,
                "source_commit": self.manifest["source_commit"], **self.counts}
