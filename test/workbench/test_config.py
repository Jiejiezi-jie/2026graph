from pathlib import Path

from app.config import Settings


def test_settings_fix_workspace_and_graph_storage(tmp_path: Path) -> None:
    settings = Settings(app_root=tmp_path, deepseek_api_key="test-secret")

    assert settings.active_dir == tmp_path / "data" / "active"
    assert settings.workspace_dir == tmp_path / "data" / "workspace"
    assert settings.graph_storage == "NetworkXStorage"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dim == 384
