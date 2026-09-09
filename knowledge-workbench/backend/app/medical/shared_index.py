"""PathRAG runtime snapshot of the LightRAG index; no extraction or embedding."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

FILES = (
    "graph_chunk_entity_relation.graphml", "kv_store_full_docs.json",
    "kv_store_text_chunks.json", "vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json",
)


def prepare_pathrag_index(source: Path, cache_root: Path) -> Path:
    # Separate mutable caches so PathRAG cannot overwrite LightRAG's live stores.
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in FILES}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    target = cache_root / identity
    marker = target / "shared_index.json"
    if marker.is_file() and json.loads(marker.read_text()) == hashes:
        if all((target / name).is_file() and
               hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items()):
            return target
        raise ValueError("Shared PathRAG index changed; restore the derived runtime snapshot before querying.")
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shared-index-", dir=cache_root) as temporary:
        staging = Path(temporary) / "index"
        staging.mkdir()
        for name in FILES:
            shutil.copyfile(source / name, staging / name)
        (staging / "shared_index.json").write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")
        staging.rename(target)
    return target
