import importlib.util
import json
import hashlib
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[2] / 'exports/Novel-4128-runtime/restore.py'
spec = importlib.util.spec_from_file_location('restore_runtime', script)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def fixture_bundle(tmp_path):
    bundle = tmp_path / 'bundle'
    checks = {}
    for rel in ['data/active/corpus.json', 'data/workspace/manifest.json']:
        p = bundle / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}', encoding='utf-8')
        checks[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    (bundle / 'checksums.json').write_text(json.dumps(checks), encoding='utf-8')
    return bundle


def test_restore_copies_verified_data_and_refuses_overwrite(tmp_path):
    bundle = fixture_bundle(tmp_path)
    target = tmp_path / 'project'
    runtime.restore(bundle, target)
    assert (target / 'data/active/corpus.json').read_text() == '{}'
    with pytest.raises(ValueError, match='already exists'):
        runtime.restore(bundle, target)


def test_corruption_is_detected_before_copy(tmp_path):
    bundle = fixture_bundle(tmp_path)
    (bundle / 'data/active/corpus.json').write_text('changed')
    target = tmp_path / 'project'
    with pytest.raises(ValueError, match='Checksum'):
        runtime.restore(bundle, target)
    assert not target.exists()


def test_manifest_cannot_escape_bundle(tmp_path):
    bundle = fixture_bundle(tmp_path)
    (bundle / 'checksums.json').write_text(json.dumps({'../outside': 'x'}))
    with pytest.raises(ValueError, match='Unsafe'):
        runtime.verify(bundle)
