import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.domain.errors import AppError
from app.domain.models import CorpusDocument
from app.services.dataset_service import corpus_fingerprint
from app.services.rag_lifecycle import finalize_rag


WORKSPACE_SCHEMA_VERSION = 1
GRAPH_FILE_NAME = "graph_chunk_entity_relation.graphml"


@dataclass(frozen=True)
class IndexOutcome:
    action: Literal["built", "reused"]


class WorkspaceService:
    def __init__(
        self,
        workspace_dir: Path,
        factory: Any,
        runtime_fingerprint: str,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.factory = factory
        self.runtime_fingerprint = runtime_fingerprint
        self._lock = asyncio.Lock()

    async def ensure_ready(self, document: CorpusDocument) -> IndexOutcome:
        async with self._lock:
            corpus_digest = corpus_fingerprint(document)
            existing = self._load_manifest()
            if existing and existing.get("status") == "ready":
                if (
                    existing.get("corpus_fingerprint") != corpus_digest
                    or existing.get("runtime_fingerprint")
                    != self.runtime_fingerprint
                ):
                    raise AppError(
                        "INDEX_CONFIG_MISMATCH",
                        "The existing workspace was built for different data or settings",
                    )
                self._verify_graph()
                self._verify_persisted_index()
                return IndexOutcome(action="reused")

            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            started_at = self._now()
            started = time.perf_counter()
            base_manifest = {
                "schema_version": WORKSPACE_SCHEMA_VERSION,
                "corpus_fingerprint": corpus_digest,
                "runtime_fingerprint": self.runtime_fingerprint,
                "corpus_name": document.corpus_name,
                "started_at": started_at,
            }
            self._write_manifest({**base_manifest, "status": "building"})

            try:
                rag = await self.factory.create()
                try:
                    await rag.ainsert(document.context)
                except BaseException:
                    # Persist partial work on failure without masking its cause.
                    try:
                        await finalize_rag(rag)
                    except Exception:
                        pass
                    raise
                await finalize_rag(rag)
                self._verify_graph()
                self._verify_persisted_index()
            except Exception as exc:
                self._write_manifest(
                    {
                        **base_manifest,
                        "status": "failed",
                        "completed_at": self._now(),
                        "latency_ms": round(
                            (time.perf_counter() - started) * 1000,
                            3,
                        ),
                        "error_type": type(exc).__name__,
                    }
                )
                raise

            self._write_manifest(
                {
                    **base_manifest,
                    "status": "ready",
                    "completed_at": self._now(),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            return IndexOutcome(action="built")

    def get_status(self, document: CorpusDocument | None) -> dict[str, Any]:
        empty = {
            "status": "not_built",
            "reusable": False,
            "corpus_name": document.corpus_name if document else None,
            "started_at": None,
            "completed_at": None,
            "latency_ms": None,
            "error_type": None,
        }
        if document is None:
            return empty

        manifest = self._load_manifest()
        if manifest is None:
            return empty

        status = manifest.get("status")
        if status not in {"building", "ready", "failed", "interrupted"}:
            status = "failed"
        reusable = (
            status == "ready"
            and manifest.get("corpus_fingerprint") == corpus_fingerprint(document)
            and manifest.get("runtime_fingerprint") == self.runtime_fingerprint
        )
        if reusable:
            try:
                self._verify_graph()
                self._verify_persisted_index()
            except AppError:
                reusable = False

        return {
            "status": status,
            "reusable": reusable,
            "corpus_name": manifest.get("corpus_name", document.corpus_name),
            "started_at": manifest.get("started_at"),
            "completed_at": manifest.get("completed_at"),
            "latency_ms": manifest.get("latency_ms"),
            "error_type": manifest.get("error_type"),
        }

    def _verify_graph(self) -> None:
        graph_path = self.workspace_dir / GRAPH_FILE_NAME
        if not graph_path.is_file() or graph_path.stat().st_size == 0:
            raise AppError(
                "INDEX_INCOMPLETE",
                "The LightRAG graph file is missing or empty",
            )

    def _verify_persisted_index(self) -> None:
        """ainsert() may persist FAILED status without raising an exception."""
        try:
            status = json.loads((self.workspace_dir / "kv_store_doc_status.json").read_text(encoding="utf-8"))
            if not isinstance(status, dict) or not status or any(
                not isinstance(row, dict) or row.get("status") != "processed"
                for row in status.values()
            ):
                raise ValueError("Document not processed")
            chunks = json.loads((self.workspace_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
            if not isinstance(chunks, dict) or not chunks:
                raise ValueError("No chunks")
            for filename in ("vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json"):
                vectors = json.loads((self.workspace_dir / filename).read_text(encoding="utf-8"))
                if not isinstance(vectors, dict) or not isinstance(vectors.get("data"), list):
                    raise ValueError("Invalid vector store")
                if filename == "vdb_chunks.json" and not vectors["data"]:
                    raise ValueError("No chunk vectors")
        except (OSError, UnicodeError, ValueError) as exc:
            raise AppError("INDEX_INCOMPLETE", "文档处理未完成，或 chunk / 向量文件缺失；不能将索引标记为就绪。") from exc

    def _load_manifest(self) -> dict[str, Any] | None:
        path = self.workspace_dir / "manifest.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AppError(
                "INVALID_INDEX_MANIFEST",
                "The workspace manifest cannot be read",
            ) from exc
        if not isinstance(payload, dict):
            raise AppError(
                "INVALID_INDEX_MANIFEST",
                "The workspace manifest is invalid",
            )
        return payload

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        path = self.workspace_dir / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            raise AppError(
                "INDEX_MANIFEST_WRITE_FAILED",
                "The workspace manifest could not be saved",
            ) from exc

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
