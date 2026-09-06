import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.adapters.graphrag_bench import GraphRAGBenchLoader
from app.api.dependencies import AppContext
from app.config import Settings
from app.main import create_app
from app.services.dataset_service import DatasetService, corpus_fingerprint
from app.services.workspace_service import WorkspaceService


class NeverCreateFactory:
    async def create(self):
        raise AssertionError("read-only status routes must not create LightRAG")


def make_client(tmp_path: Path) -> tuple[TestClient, DatasetService, WorkspaceService]:
    settings = Settings(app_root=tmp_path, deepseek_api_key="test-secret")
    dataset_service = DatasetService(GraphRAGBenchLoader(), settings.active_dir)
    workspace_service = WorkspaceService(
        settings.workspace_dir,
        NeverCreateFactory(),
        "runtime-test",
    )
    context = AppContext(
        settings=settings,
        dataset_service=dataset_service,
        workspace_service=workspace_service,
    )
    return TestClient(create_app(context)), dataset_service, workspace_service


def test_health_reports_configuration_without_secret(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "api_key_configured": True,
        "llm_model": "deepseek-v4-flash",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "graph_storage": "NetworkXStorage",
    }
    assert "test-secret" not in response.text


def test_missing_dataset_and_index_have_explicit_empty_status(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)

    dataset_response = client.get("/api/dataset/status")
    index_response = client.get("/api/index/status")

    assert dataset_response.status_code == 200
    assert dataset_response.json() == {"selected": False, "dataset": None}
    assert index_response.status_code == 200
    assert index_response.json() == {
        "status": "not_built",
        "reusable": False,
        "corpus_name": None,
        "started_at": None,
        "completed_at": None,
        "latency_ms": None,
        "error_type": None,
    }


def test_matching_ready_manifest_is_reported_as_reusable(tmp_path: Path) -> None:
    client, dataset_service, workspace_service = make_client(tmp_path)
    corpus_path = tmp_path / "novel.json"
    corpus_path.write_text(
        json.dumps([{"corpus_name": "Novel-test", "context": "one two"}]),
        encoding="utf-8",
    )
    document = dataset_service.select_shortest(corpus_path, "novel")
    workspace_service.workspace_dir.mkdir(parents=True)
    (workspace_service.workspace_dir / "graph_chunk_entity_relation.graphml").write_text(
        "<graphml />",
        encoding="utf-8",
    )
    (workspace_service.workspace_dir / "kv_store_doc_status.json").write_text('{"doc":{"status":"processed"}}', encoding="utf-8")
    (workspace_service.workspace_dir / "kv_store_text_chunks.json").write_text('{"chunk":{"content":"one two"}}', encoding="utf-8")
    for name in ("vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json"):
        (workspace_service.workspace_dir / name).write_text('{"data":[{"__id__":"one"}]}', encoding="utf-8")
    (workspace_service.workspace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ready",
                "corpus_name": document.corpus_name,
                "corpus_fingerprint": corpus_fingerprint(document),
                "runtime_fingerprint": "runtime-test",
                "started_at": "2026-09-04T00:00:00+00:00",
                "completed_at": "2026-09-04T00:01:00+00:00",
                "latency_ms": 60000.0,
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/api/index/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "reusable": True,
        "corpus_name": "Novel-test",
        "started_at": "2026-09-04T00:00:00+00:00",
        "completed_at": "2026-09-04T00:01:00+00:00",
        "latency_ms": 60000.0,
        "error_type": None,
    }
