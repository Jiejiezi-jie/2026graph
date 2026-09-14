"""Export a Medical runtime bundle from a local Git ref, without checkout or API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import tarfile


def export_bundle(repo: Path, ref: str, output: Path):
    repo, output = repo.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Medical bundle: {output}")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    commit = git("rev-parse", "--verify", ref + "^{commit}").decode().strip()
    names = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
    selected = {}
    for name in names:
        if name.startswith("results_api/indexes/"):
            selected[name] = "indexes/" + name.removeprefix("results_api/indexes/")
        elif name.startswith("src/official_backends/") and name.endswith(".py"):
            selected[name] = "vendor/official_backends/" + name.rsplit("/", 1)[-1]
        elif name == "results_api/p1/analysis/router.joblib":
            selected[name] = "router.joblib"
        elif name.upper() in {"LICENSE", "LICENSE.MD", "LICENSE.TXT", "NOTICE"}:
            selected[name] = "vendor/" + name
    if not any(n.startswith("indexes/") for n in selected.values()) or "router.joblib" not in selected.values():
        raise ValueError("This revision does not include Medical indexes and the trained router.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build in a sibling temp directory. A failed export never leaves a launchable partial bundle.
    with tempfile.TemporaryDirectory(prefix="medical-import-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        archive = Path(temporary) / "snapshot.tar"
        subprocess.run(["git", "-C", str(repo), "archive", "-o", str(archive), commit, *selected], check=True)
        files = {}
        with tarfile.open(archive) as source:
            for member in source:
                if member.name not in selected:
                    continue
                relative = selected[member.name]
                target = staging / relative
                if not member.isfile() or not target.resolve().is_relative_to(staging.resolve()):
                    raise ValueError("Unexpected archive entry")
                payload = source.extractfile(member).read()
                if payload.startswith(b"version https://git-lfs.github.com/spec/"):
                    raise ValueError(f"Git LFS pointer instead of actual asset: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                files[relative] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        manifest = {
            "schema_version": 1, "source_commit": commit,
            "source_repository": "https://github.com/Jiejiezi-jie/2026graph",
            "source_ref": ref, "corpus_name": "Medical", "embedding_max_length": 2048,
            "files": files,
        }
        (staging / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ref", default="origin/main")
    parser.add_argument("--legacy-router", action="store_true", help="保留旧 TF-IDF 选择头，仅用于旧提交")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "backend/data/medical")
    args = parser.parse_args()
    manifest = export_bundle(args.repo, args.ref, args.output)
    if not args.legacy_router:
        from install_medical_router import install_router
        install_router(args.repo, args.ref, args.output)
    print(f"Exported {len(manifest['files'])} files at commit {manifest['source_commit']}")
    print(f"Medical bundle: {args.output.resolve()}")
    print("Next: run scripts/serve_web.py --medical-bundle PATH --check")


if __name__ == "__main__":
    main()
