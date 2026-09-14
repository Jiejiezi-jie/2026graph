"""Fetch the pinned upstream LightRAG and PathRAG source trees."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


LIGHTRAG_URL = "https://github.com/HKUDS/LightRAG.git"
LIGHTRAG_COMMIT = "28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c"
PATHRAG_URL = "https://github.com/BUPT-GAMMA/PathRAG.git"
PATHRAG_COMMIT = "32567bfc93605b8393996d5fa9ccdc0edbb865b2"
BENCHMARK_URL = "https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git"
BENCHMARK_COMMIT = "fdbab5959b18c96532580877ffe27d112bccc0ec"


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(args), cwd=cwd, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def vendored_commit(path: Path) -> str | None:
    """Revision declared by a pruned upstream copy shipped without `.git`.

    `deps/LightRAG` and `deps/PathRAG` are upstream snapshots committed to this
    repository, so the revision they were taken from is recorded in their
    `UPSTREAM_COMMIT` marker rather than in Git metadata.
    """
    marker = path / "UPSTREAM_COMMIT"
    if not marker.is_file():
        return None
    try:
        for line in marker.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        return None
    return None


def ensure_repo(path: Path, url: str, commit: str) -> None:
    if (path / ".git").exists():
        actual_url = run("git", "config", "--get", "remote.origin.url", cwd=path)
        if actual_url.rstrip("/") != url.rstrip("/"):
            raise RuntimeError(f"Unexpected remote for {path}: {actual_url}")
    elif path.is_dir() and any(path.iterdir()):
        # Vendored copy: cloning here would fail on the non-empty directory, and
        # there is nothing to pin, so the marker is the only revision evidence.
        marked = vendored_commit(path)
        if marked == commit:
            print(f"{path}: {marked} (vendored copy, no Git checkout to pin)")
            return
        detail = f"UPSTREAM_COMMIT says {marked}" if marked else "no UPSTREAM_COMMIT marker"
        raise RuntimeError(
            f"{path} is a vendored copy without .git and does not declare {commit}"
            f" ({detail}); remove the directory or use another --root to clone instead"
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", url, str(path))
    run("git", "fetch", "--depth", "1", "origin", commit, cwd=path)
    run("git", "checkout", "--detach", commit, cwd=path)
    actual = run("git", "rev-parse", "HEAD", cwd=path)
    if actual != commit:
        raise RuntimeError(f"{path} resolved to {actual}, expected {commit}")
    print(f"{path}: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("deps"))
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/vendor/GraphRAG-Benchmark"),
    )
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    ensure_repo(args.root / "LightRAG", LIGHTRAG_URL, LIGHTRAG_COMMIT)
    ensure_repo(args.root / "PathRAG", PATHRAG_URL, PATHRAG_COMMIT)
    if not args.skip_benchmark:
        ensure_repo(args.benchmark_dir, BENCHMARK_URL, BENCHMARK_COMMIT)


if __name__ == "__main__":
    main()
