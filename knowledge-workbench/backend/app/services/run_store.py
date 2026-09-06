import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager


class RunStore:
    """Local run history; never saves credentials or provider request objects."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            with conn:
                conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT, payload TEXT)")
                yield conn
        finally:
            conn.close()

    def save(self, result: dict) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO runs VALUES (?, ?, ?)",
                         (result["id"], result["created_at"], json.dumps(result, ensure_ascii=False)))
            # Bound local history; evidence can be large.
            conn.execute("DELETE FROM runs WHERE id NOT IN (SELECT id FROM runs ORDER BY created_at DESC LIMIT 100)")

    def list_recent(self) -> list[dict]:
        if not self.path.is_file():
            return []
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM runs ORDER BY created_at DESC LIMIT 20").fetchall()
        keys = ("id", "created_at", "query", "method_id", "options", "status", "latency_ms", "error")
        return [{k: data.get(k) for k in keys} for (raw,) in rows for data in [json.loads(raw)]]

    def get(self, run_id: str) -> dict | None:
        if not self.path.is_file():
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None
