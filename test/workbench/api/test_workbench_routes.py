import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.adapters.graphrag_bench import GraphRAGBenchLoader
from app.api.dependencies import AppContext
from app.config import Settings
from app.main import create_app
from app.retrieval.contracts import MethodDescriptor, RetrievalResult
from app.retrieval.registry import RetrievalRegistry
from app.services.dataset_service import DatasetService
from app.services.workspace_service import WorkspaceService


class ExternalPlugin:
    descriptor = MethodDescriptor(id="my-retrieval", name="自定义检索", description="Independent")
    async def retrieve(self, request):
        return RetrievalResult(context_text="Retrieved text", chunks=[{"content": "Retrieved text"}])


class Generator:
    async def generate(self, query, result):
        return "Grounded answer"


class NeverCreateFactory:
    async def create(self):
        raise AssertionError("Should not instantiate LightRAG")


def client_for(tmp_path):
    settings = Settings(app_root=tmp_path, benchmark_root=tmp_path / "benchmark", _env_file=None)
    context = AppContext(settings, DatasetService(GraphRAGBenchLoader(), settings.active_dir),
                         WorkspaceService(settings.workspace_dir, NeverCreateFactory(), "test"))
    return TestClient(create_app(context, registry=RetrievalRegistry([ExternalPlugin()]), generator=Generator()))


def test_custom_plugin_available_over_http_without_lightrag_workspace(tmp_path):
    with client_for(tmp_path) as client:
        assert client.get("/api/retrieval/methods").json()["methods"][0]["id"] == "my-retrieval"
        response = client.post("/api/query", json={"query": "question", "method_id": "my-retrieval"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Grounded answer"
        assert client.get("/api/runs").json()["runs"][0]["id"] == data["id"]
        assert client.get("/api/runs/" + data["id"]).json()["retrieval"]["chunks"]
        assert client.get("/api/runs/not-found").status_code == 404


def test_streaming_query_keeps_legacy_results_and_history(tmp_path):
    with client_for(tmp_path) as client:
        response = client.post("/api/query/stream", json={"query": "question", "method_id": "my-retrieval"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.text.splitlines()]
        assert [event["type"] for event in events] == ["retrieval", "result"]
        result = events[-1]["run"]
        assert result["answer"] == "Grounded answer"
        assert client.get("/api/runs").json()["runs"][0]["id"] == result["id"]
        assert client.post("/api/query/stream", json={"query": " "}).status_code == 422
        assert client.post("/api/query/stream", json={"query": "x", "method_id": "missing"}).status_code == 400


def test_only_imported_readonly_graph_can_be_read_during_query(tmp_path):
    from unittest.mock import Mock
    with client_for(tmp_path) as client:
        runtime = client.app.state.runtime
        runtime.lock = Mock(locked=lambda: True)
        assert client.get("/api/graph").status_code == 409
        runtime.imported = True
        runtime.status = lambda: {"reusable": True}
        runtime.preview = lambda method: {"nodes": [], "method": method}
        assert client.get("/api/graph?method_id=pathrag").json() == {"nodes": [], "method": "pathrag"}
        runtime.imported = False


def test_streaming_query_keeps_legacy_results_and_history(tmp_path):
    with client_for(tmp_path) as client:
        response = client.post("/api/query/stream", json={"query": "question", "method_id": "my-retrieval"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.text.splitlines()]
        assert [event["type"] for event in events] == ["retrieval", "result"]
        result = events[-1]["run"]
        assert result["answer"] == "Grounded answer"
        assert client.get("/api/runs").json()["runs"][0]["id"] == result["id"]
        assert client.post("/api/query/stream", json={"query": " "}).status_code == 422
        assert client.post("/api/query/stream", json={"query": "x", "method_id": "missing"}).status_code == 400


def test_only_imported_readonly_graph_can_be_read_during_query(tmp_path):
    from unittest.mock import Mock
    with client_for(tmp_path) as client:
        runtime = client.app.state.runtime
        runtime.lock = Mock(locked=lambda: True)
        assert client.get("/api/graph").status_code == 409
        runtime.imported = True
        runtime.status = lambda: {"reusable": True}
        runtime.preview = lambda method: {"nodes": [], "method": method}
        assert client.get("/api/graph?method_id=pathrag").json() == {"nodes": [], "method": "pathrag"}
        runtime.imported = False


def test_query_validation_unknown_method_and_no_dataset(tmp_path):
    with client_for(tmp_path) as client:
        for payload in ({"query": " "}, {"query": "a", "top_k": 0}, {"query": "a", "top_k": True}):
            assert client.post("/api/query", json=payload).status_code == 422
        assert client.post("/api/query", json={"query": "a", "method_id": "absent"}).status_code == 400
        assert client.post("/api/index/build", json={}).json()["error"]["code"] == "NO_DATASET"
        assert client.get("/api/graph").status_code == 409


def test_selection_uses_allowed_dataset_paths_only(tmp_path):
    path = tmp_path / "benchmark" / "Datasets" / "Corpus"
    path.mkdir(parents=True)
    (path / "novel.json").write_text(json.dumps([
        {"corpus_name": "long", "context": "one two three"},
        {"corpus_name": "short", "context": "one"},
    ]), encoding="utf-8")
    with client_for(tmp_path) as client:
        assert client.post("/api/dataset/select", json={"subset": "../../secret"}).status_code == 422
        response = client.post("/api/dataset/select", json={"subset": "novel"})
        assert response.json()["dataset"]["corpus_name"] == "short"
        assert client.get("/api/index/status").json()["status"] == "not_built"


def test_stale_build_manifest_shown_as_interrupted(tmp_path):
    from app.domain.models import CorpusDocument
    active = tmp_path / "data" / "active"
    active.mkdir(parents=True)
    document = CorpusDocument(corpus_name="one", context="text", subset="novel", character_count=4, word_count=1)
    (active / "corpus.json").write_text(document.model_dump_json(), encoding="utf-8")
    workspace = tmp_path / "data" / "workspace"
    workspace.mkdir()
    (workspace / "manifest.json").write_text('{"status":"building"}', encoding="utf-8")
    with client_for(tmp_path) as client:
        assert client.get("/api/index/status").json()["status"] == "interrupted"


def test_corrupt_manifest_is_reported_as_failed_not_http_error(tmp_path):
    from app.domain.models import CorpusDocument
    active = tmp_path / "data" / "active"
    active.mkdir(parents=True)
    document = CorpusDocument(corpus_name="one", context="text", subset="novel", character_count=4, word_count=1)
    (active / "corpus.json").write_text(document.model_dump_json(), encoding="utf-8")
    workspace = tmp_path / "data" / "workspace"
    workspace.mkdir()
    (workspace / "manifest.json").write_text("broken JSON", encoding="utf-8")
    with client_for(tmp_path) as client:
        response = client.get("/api/index/status")
        assert response.status_code == 200
        assert response.json()["status"] == "failed"
        assert response.json()["error_type"] == "INVALID_INDEX_MANIFEST"
