import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def test_export_uses_committed_revision_not_working_files(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_medical.py"
    assert script.exists(), "Medical exporter has not been implemented"
    spec = importlib.util.spec_from_file_location("prepare_medical", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    git("init")
    for name in ["src/official_backends/__init__.py", "src/official_backends/model_client.py",
                 "results_api/indexes/vector/index.json", "results_api/p1/analysis/router.joblib",
                 "configs/official_api.json"]:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"test": "committed"}')
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-m", "fixture")
    (repo / "results_api/indexes/vector/index.json").write_text('{"test": "uncommitted"}')
    output = tmp_path / "bundle"
    module.export_bundle(repo, "HEAD", output)
    assert json.loads((output / "indexes/vector/index.json").read_text()) == {"test": "committed"}
    assert not (output / "configs").exists()
    manifest = json.loads((output / "bundle.json").read_text())
    assert len(manifest["source_commit"]) == 40
    assert manifest["files"]["indexes/vector/index.json"]["sha256"]
    with pytest.raises(FileExistsError):
        module.export_bundle(repo, "HEAD", output)
