import json
from pathlib import Path
from typing import Any, Iterable

from app.domain.errors import AppError
from app.domain.models import CorpusDocument


class GraphRAGBenchLoader:
    def load_documents(self, path: Path, subset: str) -> list[CorpusDocument]:
        path = Path(path)
        if not path.is_file():
            raise AppError("CORPUS_NOT_FOUND", "The corpus file does not exist")

        suffix = path.suffix.lower()
        if suffix == ".json":
            rows = self._load_json(path)
        elif suffix == ".parquet":
            rows = self._load_parquet(path)
        else:
            raise AppError(
                "UNSUPPORTED_CORPUS",
                "Only GraphRAG-Bench JSON and Parquet corpora are supported",
            )

        documents = [self._to_document(row, subset) for row in rows]
        if not documents:
            raise AppError("INVALID_CORPUS", "The corpus contains no documents")
        return documents

    @staticmethod
    def _load_json(path: Path) -> Iterable[Any]:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AppError("INVALID_CORPUS", "The corpus JSON cannot be read") from exc
        if not isinstance(rows, list):
            raise AppError("INVALID_CORPUS", "The corpus JSON must contain an array")
        return rows

    @staticmethod
    def _load_parquet(path: Path) -> Iterable[Any]:
        try:
            from datasets import load_dataset

            return load_dataset(
                "parquet",
                data_files=str(path),
                split="train",
            )
        except Exception as exc:
            raise AppError("INVALID_CORPUS", "The corpus Parquet file cannot be read") from exc

    @staticmethod
    def _to_document(row: Any, subset: str) -> CorpusDocument:
        if not isinstance(row, dict):
            raise AppError("INVALID_CORPUS", "Each corpus row must be an object")
        corpus_name = row.get("corpus_name")
        context = row.get("context")
        if not isinstance(corpus_name, str) or not corpus_name.strip():
            raise AppError("INVALID_CORPUS", "A corpus row has no valid corpus_name")
        if not isinstance(context, str) or not context.strip():
            raise AppError("INVALID_CORPUS", "A corpus row has no valid context")
        return CorpusDocument(
            corpus_name=corpus_name,
            context=context,
            subset=subset,
            character_count=len(context),
            word_count=len(context.split()),
        )
