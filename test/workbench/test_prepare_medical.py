import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "src/backend/scripts/prepare_medical.py"

CURRENT_FILES = [
    "src/backend/__init__.py",
    "src/backend/common/base.py",
    "src/backend/common/model_client.py",
    "src/backend/vector/vector_backend.py",
    "result/api/indexes/vector/index.json",
    "result/api/p1/analysis/router.joblib",
    "configs/official_api.json",
]
# Pre-restructure names, so bundles can still be exported from historical commits.
LEGACY_FILES = [
    "src/official_backends/__init__.py",
    "src/official_backends/model_client.py",
    "results_api/indexes/vector/index.json",
    "results_api/p1/analysis/router.joblib",
    "configs/official_api.json",
]


def load_exporter():
    assert SCRIPT.exists(), "Medical exporter has not been implemented"
    spec = importlib.util.spec_from_file_location("prepare_medical", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commit_fixture(repo, names, marker='{"test": "committed"}'):
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    git("init")
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker)
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-m", "fixture")
    return git


def test_export_uses_committed_revision_not_working_files(tmp_path):
    module = load_exporter()
    repo = tmp_path / "repo"
    commit_fixture(repo, CURRENT_FILES)
    (repo / "result/api/indexes/vector/index.json").write_text('{"test": "uncommitted"}')
    output = tmp_path / "bundle"
    module.export_bundle(repo, "HEAD", output)
    assert json.loads((output / "indexes/vector/index.json").read_text()) == {"test": "committed"}
    assert not (output / "configs").exists()
    manifest = json.loads((output / "bundle.json").read_text())
    assert len(manifest["source_commit"]) == 40
    assert manifest["files"]["indexes/vector/index.json"]["sha256"]
    with pytest.raises(FileExistsError):
        module.export_bundle(repo, "HEAD", output)


def test_export_still_reads_pre_restructure_revisions(tmp_path):
    module = load_exporter()
    repo = tmp_path / "repo"
    commit_fixture(repo, LEGACY_FILES)
    output = tmp_path / "bundle"
    module.export_bundle(repo, "HEAD", output)
    assert (output / "indexes/vector/index.json").exists()
    assert (output / "router.joblib").exists()
    assert (output / "vendor/official_backends/model_client.py").exists()
