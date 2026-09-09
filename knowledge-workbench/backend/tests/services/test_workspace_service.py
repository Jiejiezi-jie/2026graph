import json
from pathlib import Path

import pytest

from app.config import Settings
from app.domain.errors import AppError
from app.domain.models import CorpusDocument
from app.services.lightrag_factory import LightRAGFactory
from app.services.workspace_service import WorkspaceService


class FakeRAG:
    def __init__(
        self,
        factory: "FakeFactory",
        *,
        finalize_error: Exception | None = None,
        write_graph: bool = True,
    ) -> None:
        self.factory = factory
        self.finalize_error = finalize_error
        self.write_graph = write_graph

    async def ainsert(self, text: str) -> None:
        self.factory.inserted.append(text)

    async def finalize_storages(self) -> None:
        self.factory.finalized += 1
        if self.finalize_error is not None:
            raise self.finalize_error
        if self.write_graph:
            self.factory.workspace_dir.mkdir(parents=True, exist_ok=True)
            (self.factory.workspace_dir / "graph_chunk_entity_relation.graphml").write_text(
                "<graphml />",
                encoding="utf-8",
            )
            (self.factory.workspace_dir / "kv_store_doc_status.json").write_text('{"doc":{"status":"processed"}}', encoding="utf-8")
            (self.factory.workspace_dir / "kv_store_text_chunks.json").write_text('{"chunk":{"content":"one two"}}', encoding="utf-8")
            for name in ("vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json"):
                (self.factory.workspace_dir / name).write_text('{"data":[{"__id__":"one"}]}', encoding="utf-8")


class FakeFactory:
    def __init__(
        self,
        workspace_dir: Path,
        *,
        finalize_error: Exception | None = None,
        write_graph: bool = True,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.finalize_error = finalize_error
        self.write_graph = write_graph
        self.create_count = 0
        self.inserted: list[str] = []
        self.finalized = 0

    async def create(self) -> FakeRAG:
        self.create_count += 1
        return FakeRAG(
            self,
            finalize_error=self.finalize_error,
            write_graph=self.write_graph,
        )


def make_document(text: str) -> CorpusDocument:
    return CorpusDocument(
        corpus_name="Novel-test",
        context=text,
        subset="novel",
        character_count=len(text),
        word_count=len(text.split()),
    )


@pytest.mark.asyncio
async def test_second_call_reuses_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    factory = FakeFactory(workspace)
    service = WorkspaceService(workspace, factory, "runtime-1")

    first = await service.ensure_ready(make_document("one two"))
    second = await service.ensure_ready(make_document("one two"))

    assert first.action == "built"
    assert second.action == "reused"
    assert factory.create_count == 1
    assert factory.inserted == ["one two"]
    manifest = json.loads((workspace / "manifest.json").read_text())
    assert manifest["status"] == "ready"


@pytest.mark.asyncio
async def test_changed_document_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    service = WorkspaceService(workspace, FakeFactory(workspace), "runtime-1")
    await service.ensure_ready(make_document("first text"))

    with pytest.raises(AppError) as error:
        await service.ensure_ready(make_document("changed text"))

    assert error.value.code == "INDEX_CONFIG_MISMATCH"


@pytest.mark.asyncio
async def test_finalize_failure_writes_failed_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    factory = FakeFactory(workspace, finalize_error=RuntimeError("disk detail"))
    service = WorkspaceService(workspace, factory, "runtime-1")

    with pytest.raises(RuntimeError, match="disk detail"):
        await service.ensure_ready(make_document("one two"))

    manifest = json.loads((workspace / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"
    assert "disk detail" not in json.dumps(manifest)


@pytest.mark.asyncio
async def test_missing_graph_file_writes_failed_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    factory = FakeFactory(workspace, write_graph=False)
    service = WorkspaceService(workspace, factory, "runtime-1")

    with pytest.raises(AppError) as error:
        await service.ensure_ready(make_document("one two"))

    assert error.value.code == "INDEX_INCOMPLETE"
    manifest = json.loads((workspace / "manifest.json").read_text())
    assert manifest["status"] == "failed"


@pytest.mark.asyncio
async def test_factory_rejects_empty_api_key_before_loading_models(
    tmp_path: Path,
) -> None:
    settings = Settings(app_root=tmp_path, deepseek_api_key="")

    with pytest.raises(AppError) as error:
        await LightRAGFactory(settings).create()

    assert error.value.code == "MISSING_API_KEY"


def test_runtime_fingerprint_changes_with_model_configuration(tmp_path: Path) -> None:
    first = LightRAGFactory(
        Settings(app_root=tmp_path, deepseek_api_key="secret", llm_model="model-a")
    )
    second = LightRAGFactory(
        Settings(app_root=tmp_path, deepseek_api_key="secret", llm_model="model-b")
    )

    assert first.runtime_fingerprint() != second.runtime_fingerprint()


@pytest.mark.asyncio
async def test_silent_document_failure_is_not_marked_ready(tmp_path: Path):
    workspace = tmp_path / "workspace"
    factory = FakeFactory(workspace)
    rag = FakeRAG(factory)
    original_finalize = rag.finalize_storages
    async def finalize():
        await original_finalize()
        (workspace / "kv_store_doc_status.json").write_text(
            '{"doc-test":{"status":"failed","error_msg":"secret provider details"}}', encoding="utf-8")
    rag.finalize_storages = finalize
    async def create():
        return rag
    factory.create = create
    service = WorkspaceService(workspace, factory, "test")
    with pytest.raises(AppError) as caught:
        await service.ensure_ready(make_document("one two"))
    assert caught.value.code == "INDEX_INCOMPLETE"
    assert json.loads((workspace / "manifest.json").read_text())["status"] == "failed"
    assert "secret provider" not in str(caught.value)
