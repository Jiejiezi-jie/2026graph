import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.adapters.graphrag_bench import GraphRAGBenchLoader
from app.domain.errors import AppError
from app.domain.models import CorpusDocument


ACTIVE_SCHEMA_VERSION = 1


def corpus_fingerprint(document: CorpusDocument) -> str:
    digest = hashlib.sha256()
    for value in (document.subset, document.corpus_name, document.context):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class DatasetService:
    def __init__(self, loader: GraphRAGBenchLoader, active_dir: Path) -> None:
        self.loader = loader
        self.active_dir = Path(active_dir)

    def select_shortest(self, path: Path, subset: str) -> CorpusDocument:
        documents = self.loader.load_documents(path, subset)
        selected = min(
            documents,
            key=lambda item: (item.character_count, item.corpus_name),
        )
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(
            self.active_dir / "corpus.json",
            selected.model_dump(mode="json"),
        )
        self._write_json_atomic(
            self.active_dir / "manifest.json",
            {
                "schema_version": ACTIVE_SCHEMA_VERSION,
                "source_path": str(Path(path).resolve()),
                "subset": selected.subset,
                "corpus_name": selected.corpus_name,
                "character_count": selected.character_count,
                "word_count": selected.word_count,
                "fingerprint": corpus_fingerprint(selected),
            },
        )
        return selected

    def load_active(self) -> CorpusDocument | None:
        corpus_path = self.active_dir / "corpus.json"
        if not corpus_path.is_file():
            return None
        try:
            payload = json.loads(corpus_path.read_text(encoding="utf-8"))
            return CorpusDocument.model_validate(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            raise AppError(
                "INVALID_ACTIVE_DATASET",
                "The active dataset cannot be read",
            ) from exc

    def get_status(self) -> dict[str, Any]:
        document = self.load_active()
        if document is None:
            return {"selected": False, "dataset": None}
        return {
            "selected": True,
            "dataset": {
                "corpus_name": document.corpus_name,
                "subset": document.subset,
                "character_count": document.character_count,
                "word_count": document.word_count,
            },
        }

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            raise AppError(
                "ACTIVE_DATASET_WRITE_FAILED",
                "The active dataset could not be saved",
            ) from exc
