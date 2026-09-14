import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[2] / "src/backend/app/medical/preflight.py"
MARKER = "# vendored pruned copy\n32567bfc93605b8393996d5fa9ccdc0edbb865b2\n"


def load_module():
    spec = importlib.util.spec_from_file_location("medical_preflight", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(tmp_path, marker=None, checkout=False):
    root = tmp_path / "PathRAG"
    (root / "PathRAG").mkdir(parents=True)
    (root / "PathRAG/__init__.py").write_text("")
    if marker is not None:
        (root / "UPSTREAM_COMMIT").write_text(marker)
    if checkout:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
        git("init")
        git("add", ".")
        git("-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-m", "checkout")
    return root


def test_vendored_copy_reports_the_marked_upstream_commit(tmp_path):
    preflight = load_module()
    assert preflight.pathrag_revision(make_source(tmp_path, MARKER)) == preflight.PATHRAG_COMMIT


def test_real_checkout_wins_over_its_own_marker(tmp_path):
    preflight = load_module()
    root = make_source(tmp_path, MARKER, checkout=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    assert head != preflight.PATHRAG_COMMIT
    assert preflight.pathrag_revision(root) == head


def test_different_revision_is_still_rejected(tmp_path):
    preflight = load_module()
    root = make_source(tmp_path, "0" * 40)
    settings = SimpleNamespace(medical_embedding_path=None, medical_pathrag_root=root)
    issues = [issue for issue in preflight.runtime_issues(settings) if "PathRAG" in issue]
    assert issues == ["PathRAG 源码版本与 Medical 索引要求不符。"]


def test_vendored_copy_is_accepted_without_a_checkout(tmp_path):
    preflight = load_module()
    settings = SimpleNamespace(medical_embedding_path=None,
                               medical_pathrag_root=make_source(tmp_path, MARKER))
    issues = [issue for issue in preflight.runtime_issues(settings) if "PathRAG" in issue]
    assert issues == []


def test_plain_copy_without_marker_is_reported(tmp_path):
    preflight = load_module()
    settings = SimpleNamespace(medical_embedding_path=None,
                               medical_pathrag_root=make_source(tmp_path))
    issues = [issue for issue in preflight.runtime_issues(settings) if "PathRAG" in issue]
    assert issues and "无法确认 PathRAG 固定 Git 提交" in issues[0]
