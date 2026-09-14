import pytest
from fastapi.testclient import TestClient

from app.adapters.graphrag_bench import GraphRAGBenchLoader
from app.api.dependencies import AppContext
from app.config import Settings
from app.main import create_app
from app.services.dataset_service import DatasetService
from app.services.lightrag_factory import LightRAGFactory
from app.services.workspace_service import WorkspaceService


def app_for(tmp_path):
    settings = Settings(app_root=tmp_path, deepseek_api_key=None,
                        llm_base_url="https://api.deepseek.com", _env_file=None)
    factory = LightRAGFactory(settings)
    context = AppContext(settings, DatasetService(GraphRAGBenchLoader(), settings.active_dir),
                         WorkspaceService(settings.workspace_dir, factory, factory.runtime_fingerprint()))
    return create_app(context)


def test_settings_apply_to_both_llm_consumers_without_persisting_or_returning_key(tmp_path):
    app = app_for(tmp_path)
    with TestClient(app) as client:
        original = app.state.context.workspace_service.runtime_fingerprint
        initial = client.get("/api/settings/connection")
        assert initial.status_code == 200
        assert initial.json()["base_url"] == "https://api.deepseek.com"
        assert initial.json()["api_key_configured"] is False
        response = client.post("/api/settings/connection", json={
            "api_key": "sk-ui-test-only", "base_url": "https://api.deepseek.com/",
        })
        assert response.status_code == 200
        assert response.json()["api_key_configured"] is True
        assert "sk-ui-test-only" not in response.text
        context = app.state.context
        assert context.workspace_service.factory._api_key() == "sk-ui-test-only"
        assert app.state.runtime.queries.generator.settings.deepseek_api_key.get_secret_value() == "sk-ui-test-only"
        assert context.workspace_service.runtime_fingerprint == original
        assert client.get("/api/health").json()["api_key_configured"] is True
        assert "sk-ui-test-only" not in client.get("/api/settings/connection").text
        updated = client.post("/api/settings/connection", json={"base_url": "https://proxy.example/v1"})
        assert updated.status_code == 200
        assert context.workspace_service.factory.settings.llm_base_url == "https://proxy.example/v1"
        assert app.state.runtime.queries.generator.settings.llm_base_url == "https://proxy.example/v1"
        assert context.workspace_service.runtime_fingerprint != original
        assert context.workspace_service.factory._api_key() == "sk-ui-test-only"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"sk-ui-test-only" not in path.read_bytes()
    with TestClient(app_for(tmp_path)) as restarted:
        assert restarted.get("/api/settings/connection").json()["api_key_configured"] is False
        assert restarted.get("/api/settings/connection").json()["base_url"] == "https://api.deepseek.com"


@pytest.mark.parametrize("url", ["file:///tmp/key", "ftp://example.com", "https://user:pass@example.com", "https://example.com?key=secret", "https://example.com/#fragment", "not-a-url"])
def test_rejects_invalid_urls_without_echoing_secret(tmp_path, url):
    with TestClient(app_for(tmp_path)) as client:
        response = client.post("/api/settings/connection", json={"api_key": "sk-sensitive-input", "base_url": url})
        assert response.status_code == 422
        assert "sk-sensitive-input" not in response.text
        assert client.get("/api/settings/connection").json()["api_key_configured"] is False


@pytest.mark.parametrize("secret", ["", "   ", "key\nwith-newline", {"secret": "sk-sensitive-input"}])
def test_invalid_secret_is_rejected_and_never_echoed(tmp_path, secret):
    with TestClient(app_for(tmp_path)) as client:
        response = client.post("/api/settings/connection", json={"api_key": secret, "base_url": "https://api.deepseek.com"})
        assert response.status_code == 422
        assert "sk-sensitive-input" not in response.text


def test_busy_configuration_is_rejected_atomically(tmp_path):
    app = app_for(tmp_path)
    with TestClient(app) as client:
        client.portal.call(app.state.runtime.lock.acquire)
        try:
            response = client.post("/api/settings/connection", json={"api_key": "sk-new-key", "base_url": "https://proxy.example/v1"})
            assert response.status_code == 409
            assert client.get("/api/settings/connection").json()["base_url"] == "https://api.deepseek.com"
            assert app.state.context.settings.deepseek_api_key is None
        finally:
            client.portal.call(app.state.runtime.lock.release)


def test_cross_origin_settings_write_is_rejected(tmp_path):
    with TestClient(app_for(tmp_path)) as client:
        response = client.post("/api/settings/connection", headers={"Origin": "https://unrelated.example"}, json={
            "api_key": "sk-ui-test-only", "base_url": "https://api.deepseek.com",
        })
        assert response.status_code == 403
        assert client.get("/api/settings/connection").json()["api_key_configured"] is False
