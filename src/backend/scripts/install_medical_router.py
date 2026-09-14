"""Install the frozen shared-graph BGE router from a committed Git snapshot.

Run with the backend stopped. Keeps the old artifact and manifest for rollback.
No API calls, index changes or training.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import warnings

# Current layout first, pre-restructure path second, so historical refs still work.
ROUTER_DIR_CANDIDATES = (
    "result/shared_unified_latency/p1/router_hard_weight_grid_300/sample_weighted/threshold_0.60_margin_0.10_bge_m3",
    "results_shared_lightrag_unified_latency/p1/router_hard_weight_grid_300/sample_weighted/threshold_0.60_margin_0.10_bge_m3",
)


def install_router(repo, ref, bundle):
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    repo, bundle = Path(repo).resolve(), Path(bundle).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    commit = git("rev-parse", "--verify", ref + "^{commit}").decode().strip()
    tree = set(git("ls-tree", "-r", "--name-only", commit).decode().splitlines())
    router_dir = next(
        (d for d in ROUTER_DIR_CANDIDATES if d + "/router.joblib" in tree), None
    )
    if router_dir is None:
        raise ValueError("This revision does not contain the frozen shared-graph BGE router")
    payload = git("show", commit + ":" + router_dir + "/router.joblib")
    summary = json.loads(git("show", commit + ":" + router_dir + "/summary.json"))
    if summary.get("feature") != "bge_m3" or summary.get("correctness_threshold") != .6 or summary.get("best_margin") != .1 or summary.get("mode") != "sample_weighted" or summary.get("development_count") != 276:
        raise ValueError("Unexpected router training configuration")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(io.BytesIO(payload))
    if set(model.classes_) != {"vector", "lightrag", "pathrag"} or model.n_features_in_ != 1024:
        raise ValueError("Expected three-class 1024-dimensional BGE router")
    digest = hashlib.sha256(payload).hexdigest()
    artifact = "router-bge-" + digest[:16] + ".joblib"
    manifest_path = bundle / "bundle.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    # Content-addressed artifact is complete before the active manifest switches.
    (bundle / artifact).write_bytes(payload)
    manifest.setdefault("files", {})[artifact] = {"bytes": len(payload), "sha256": digest}
    manifest["router"] = {
        "kind": "bge_m3_cls_logistic_regression", "artifact": artifact,
        "source_commit": commit, "source_path": router_dir + "/router.joblib",
        "embedding_model": "BAAI/bge-m3", "pooling": "cls", "normalize": True,
        "max_length": 512, "dimension": 1024, "weighted": True,
        "training": summary,
    }
    backup = bundle / ("bundle-before-router-" + hashlib.sha256(original).hexdigest()[:16] + ".json")
    if not backup.exists():
        backup.write_bytes(original)
    temporary = bundle / "bundle.router-update.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest["router"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ref", default="origin/main")
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1] / "data/medical")
    args = parser.parse_args()
    print(json.dumps(install_router(args.repo, args.ref, args.bundle), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
