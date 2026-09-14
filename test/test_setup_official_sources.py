"""`setup_official_sources.py` must accept the vendored deps/ copies.

The repository ships pruned LightRAG and PathRAG snapshots without `.git`, so a
fresh clone has to run the source setup without those two directories blowing
up on `git clone` into a non-empty path.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.backend.common import setup_official_sources as sources

ROOT = Path(__file__).resolve().parents[1]
MARKER = "# vendored pruned copy\n32567bfc93605b8393996d5fa9ccdc0edbb865b2\n"


class VendoredCopyTests(unittest.TestCase):
    def vendored_copy(self, root: Path, marker=MARKER) -> Path:
        path = root / "PathRAG"
        (path / "PathRAG").mkdir(parents=True)
        (path / "PathRAG/__init__.py").write_text("")
        if marker is not None:
            (path / "UPSTREAM_COMMIT").write_text(marker)
        return path

    def ensure(self, path: Path) -> None:
        sources.ensure_repo(path, sources.PATHRAG_URL, sources.PATHRAG_COMMIT)

    def test_vendored_copy_is_adopted_without_running_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(sources, "run") as run:
                self.ensure(self.vendored_copy(Path(tmp)))
            run.assert_not_called()

    def test_marker_mismatch_names_the_declared_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.vendored_copy(Path(tmp), "0" * 40)
            with self.assertRaises(RuntimeError) as caught:
                self.ensure(path)
            self.assertIn("UPSTREAM_COMMIT says " + "0" * 40, str(caught.exception))

    def test_copy_without_a_marker_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.vendored_copy(Path(tmp), None)
            with self.assertRaises(RuntimeError) as caught:
                self.ensure(path)
            self.assertIn("no UPSTREAM_COMMIT marker", str(caught.exception))

    def test_checkout_with_a_foreign_remote_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.vendored_copy(Path(tmp), None)
            subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(path), "remote", "add", "origin",
                 "https://example.invalid/elsewhere.git"],
                check=True, capture_output=True)
            with self.assertRaises(RuntimeError) as caught:
                self.ensure(path)
            self.assertIn("Unexpected remote for", str(caught.exception))


class ShippedMarkerTests(unittest.TestCase):
    def test_markers_match_the_pinned_commits(self):
        self.assertEqual(
            sources.vendored_commit(ROOT / "deps/LightRAG"), sources.LIGHTRAG_COMMIT)
        self.assertEqual(
            sources.vendored_commit(ROOT / "deps/PathRAG"), sources.PATHRAG_COMMIT)


if __name__ == "__main__":
    unittest.main()
