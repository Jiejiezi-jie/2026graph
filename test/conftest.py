"""Make the src.* packages and the workbench's app.* package importable.

Mirrors the `pythonpath` setting in pytest.ini so the suite also runs when
invoked from elsewhere (e.g. `pytest test/...` with a different rootdir).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src/backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
