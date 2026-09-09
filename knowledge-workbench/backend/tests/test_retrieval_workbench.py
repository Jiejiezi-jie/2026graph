import asyncio
import json
from types import SimpleNamespace

import networkx as nx
import pytest

from app.domain.errors import AppError
from app.retrieval.contracts import MethodDescriptor, RetrievalRequest, RetrievalResult
from app.retrieval.registry import RetrievalRegistry
from app.retrieval.lightrag import LightRAGRetriever
from app.services.graph_service import GraphService
from app.services.query_service import QueryService
from app.services.run_store import RunStore


class TextRetriever:
    descriptor = MethodDescriptor(id="text-test", name="Text only", description="Test plugin")

    async def retrieve(self, request):
        return RetrievalResult(chunks=[{"content": request.query}], context_text=request.query)


class Generator:
    async def generate(self, query, result):
        return "Answer: " + result.context_text


@pytest.mark.asyncio
async def test_new_retriever_uses_shared_generation_and_records_run(tmp_path):
    registry = RetrievalRegistry([TextRetriever()])
    service = QueryService(registry, Generator(), RunStore(tmp_path / "runs.sqlite3"), asyncio.Lock())
    response = await service.query(RetrievalRequest(query="hello", method_id="text-test"))
    assert response["answer"] == "Answer: hello"
    assert response["retrieval"]["chunks"] == [{"content": "hello"}]
    assert response["retrieval"]["graph"]["nodes"] == []
    assert service.runs.list_recent()[0]["id"] == response["id"]
    with pytest.raises(AppError, match="Unknown retrieval method"):
        await service.query(RetrievalRequest(query="hello", method_id="missing"))


@pytest.mark.asyncio
async def test_generation_failure_preserves_retrieved_evidence(tmp_path):
    class BrokenGenerator:
        async def generate(self, query, result):
            raise TimeoutError("secret-provider-details")
    service = QueryService(RetrievalRegistry([TextRetriever()]), BrokenGenerator(), RunStore(tmp_path / "runs.db"), asyncio.Lock())
    response = await service.query(RetrievalRequest(query="hello", method_id="text-test"))
    assert response["status"] == "failure"
    assert response["retrieval"]["chunks"]
    assert response["error"]["code"] == "LLM_TIMEOUT"
    assert "secret-provider-details" not in json.dumps(response)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError("secret"), {"status": "failure", "message": "secret"}])
async def test_lightrag_handles_exception_and_failure_dictionary(failure, tmp_path):
    class Rag:
        finalized = False
        async def aquery_data(self, query, param):
            assert param.top_k == param.chunk_top_k == 7
            assert not param.enable_rerank
            if isinstance(failure, Exception):
                raise failure
            return failure
        async def finalize_storages(self):
            self.finalized = True
    rag = Rag()
    class Factory:
        async def create(self):
            return rag
    workspace = SimpleNamespace(factory=Factory(), get_status=lambda doc: {"reusable": True})
    dataset = SimpleNamespace(load_active=lambda: object())
    plugin = LightRAGRetriever(dataset, workspace, GraphService(tmp_path / "absent.graphml"))
    with pytest.raises(AppError) as caught:
        await plugin.retrieve(RetrievalRequest(query="question", top_k=7))
    assert "secret" not in str(caught.value)
    assert rag.finalized


def test_graph_marks_only_retrieved_edges_and_limits_neighborhood(tmp_path):
    graph = nx.Graph()
    graph.add_edge("A", "B", description="retrieved")
    graph.add_edge("A", "C", description="context only")
    graph.add_edge("C", "D")
    path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, path)
    service = GraphService(path)
    result = service.related([{"entity_name": "A"}], [{"src_id": "A", "tgt_id": "B"}])
    nodes = {n.id: n for n in result.nodes}
    assert set(nodes) == {"A", "B", "C"}
    assert nodes["A"].retrieved and not nodes["C"].retrieved
    assert sum(e.retrieved for e in result.edges) == 1
    assert all(not e.directed for e in result.edges)
    assert len(service.preview(max_nodes=2).nodes) == 2


@pytest.mark.asyncio
async def test_busy_query_does_not_queue_or_start_another_retrieval(tmp_path):
    lock = asyncio.Lock()
    service = QueryService(RetrievalRegistry([TextRetriever()]), Generator(), RunStore(tmp_path / "runs.db"), lock)
    await lock.acquire()
    try:
        with pytest.raises(AppError) as error:
            await service.query(RetrievalRequest(query="hello", method_id="text-test"))
        assert error.value.code == "WORKSPACE_BUSY"
    finally:
        lock.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "mix", "naive"])
async def test_lightrag_success_in_all_modes_uses_retrieval_only(mode, tmp_path):
    class Rag:
        async def aquery_data(self, query, param):
            assert param.mode == mode and param.top_k == param.chunk_top_k == 5
            return {"status": "success", "data": {"entities": [], "relationships": [],
                    "chunks": [{"chunk_id": "one", "content": "evidence"}], "references": []}}
        async def aquery_llm(self, *args, **kwargs):
            raise AssertionError("Retrieval plugin must not generate an answer")
        async def finalize_storages(self):
            pass
    class Factory:
        async def create(self):
            return Rag()
    plugin = LightRAGRetriever(SimpleNamespace(load_active=lambda: object()),
                              SimpleNamespace(factory=Factory(), get_status=lambda d: {"reusable": True}),
                              GraphService(tmp_path / "missing.graphml"))
    result = await plugin.retrieve(RetrievalRequest(query="question", options={"mode": mode}))
    assert result.chunks[0]["content"] == "evidence"
    assert result.context_text
    assert result.graph.nodes == []


@pytest.mark.asyncio
async def test_vector_mode_preserves_text_evidence_without_reading_graph():
    class Rag:
        finalized = False
        async def aquery_data(self, query, param):
            assert param.mode == "naive"
            assert param.chunk_top_k == 7
            return {"status": "success", "data": {"entities": [], "relationships": [],
                    "chunks": [{"chunk_id": "c1", "content": "vector evidence"}],
                    "references": [{"reference_id": "1", "file_path": "source.txt"}]}}
        async def finalize_storages(self):
            self.finalized = True
    rag = Rag()
    class Factory:
        async def create(self):
            return rag
    class NoGraph:
        def related(self, *args):
            raise AssertionError("Pure vector mode must not load graph neighborhoods")
    plugin = LightRAGRetriever(SimpleNamespace(load_active=lambda: object()),
                              SimpleNamespace(factory=Factory(), get_status=lambda d: {"reusable": True}), NoGraph())
    result = await plugin.retrieve(RetrievalRequest(query="question", top_k=7, options={"mode": "naive"}))
    assert result.chunks == [{"chunk_id": "c1", "content": "vector evidence"}]
    assert result.references[0]["file_path"] == "source.txt"
    assert "vector evidence" in result.context_text
    assert result.graph.nodes == result.graph.edges == []
    assert not result.warnings
    assert rag.finalized


@pytest.mark.asyncio
async def test_reset_clears_real_lightrag_namespace_cache(tmp_path):
    from lightrag.kg.json_kv_impl import JsonKVStorage
    from lightrag.kg.shared_storage import initialize_share_data, finalize_share_data
    from app.services.lightrag_factory import LightRAGFactory
    finalize_share_data()
    initialize_share_data()
    try:
        old = JsonKVStorage(namespace="web_cache_regression", workspace="",
                            global_config={"working_dir": str(tmp_path / "old")}, embedding_func=None)
        await old.initialize()
        await old.upsert({"old-chunk": {"content": "old corpus"}})
        await old.finalize()
        LightRAGFactory.reset_shared_storage()
        initialize_share_data()
        new = JsonKVStorage(namespace="web_cache_regression", workspace="",
                            global_config={"working_dir": str(tmp_path / "new")}, embedding_func=None)
        await new.initialize()
        assert await new.get_by_id("old-chunk") is None
        await new.finalize()
    finally:
        finalize_share_data()


@pytest.mark.asyncio
async def test_explicit_rebuild_archives_corrupt_workspace_and_resets_cache(tmp_path):
    from pydantic import SecretStr
    from app.api.dependencies import AppContext
    from app.config import Settings
    from app.domain.models import CorpusDocument
    from app.services.web_runtime import WebRuntime
    from app.services.workspace_service import WorkspaceService
    settings = Settings(app_root=tmp_path, deepseek_api_key=SecretStr("offline-test"), _env_file=None)
    workspace = settings.workspace_dir
    workspace.mkdir(parents=True)
    (workspace / "manifest.json").write_text("broken JSON")
    (workspace / "keep.txt").write_text("preserve me")
    events = []
    class Factory:
        def reset_shared_storage(self):
            events.append("reset")
        async def create(self):
            assert events[-1] == "reset"
            events.append("create")
            return self
        async def ainsert(self, text):
            events.append("insert")
        async def finalize_storages(self):
            (workspace / "graph_chunk_entity_relation.graphml").write_text("<graphml/>")
            (workspace / "kv_store_doc_status.json").write_text('{"doc":{"status":"processed"}}')
            (workspace / "kv_store_text_chunks.json").write_text('{"chunk":{"content":"text"}}')
            for name in ("vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json"):
                (workspace / name).write_text('{"data":[{"__id__":"one"}]}')
    doc = CorpusDocument(corpus_name="one", context="text", subset="novel", character_count=4, word_count=1)
    service = WorkspaceService(workspace, Factory(), "test")
    context = AppContext(settings, SimpleNamespace(load_active=lambda: doc), service)
    runtime = WebRuntime(context)
    assert (await runtime.start_build(rebuild=True))["action"] == "started"
    await runtime.build_task
    assert runtime.status()["reusable"]
    assert events == ["reset", "create", "insert"]
    backups = list((tmp_path / "data" / "workspace_backups").iterdir())
    assert len(backups) == 1 and (backups[0] / "keep.txt").read_text() == "preserve me"
    assert (backups[0] / "manifest.json").read_text() == "broken JSON"
    assert (await runtime.start_build(rebuild=False))["action"] == "reused"
    await runtime.start_build(rebuild=True)
    await runtime.build_task
    assert events == ["reset", "create", "insert", "reset", "create", "insert"]


@pytest.mark.asyncio
async def test_queues_are_retired_even_if_storage_finalization_fails():
    from app.services.rag_lifecycle import finalize_rag
    events = []
    class Wrapper:
        async def shutdown(self, **kwargs):
            events.append("shutdown")
    wrapped = Wrapper()
    class Rag:
        _role_llm_states = {"keyword": SimpleNamespace(wrapped=wrapped)}
        embedding_func = SimpleNamespace(func=wrapped)
        async def finalize_storages(self):
            raise RuntimeError("disk failure")
    with pytest.raises(RuntimeError, match="disk failure"):
        await finalize_rag(Rag())
    assert events == ["shutdown"]
