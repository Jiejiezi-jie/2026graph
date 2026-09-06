import json
from pathlib import Path

import pandas as pd
import pytest

from app.adapters.graphrag_bench import GraphRAGBenchLoader
from app.domain.errors import AppError


def test_loads_json_corpus_rows(tmp_path: Path) -> None:
    path = tmp_path / "novel.json"
    path.write_text(
        json.dumps([{"corpus_name": "Novel-1", "context": "one two"}]),
        encoding="utf-8",
    )

    documents = GraphRAGBenchLoader().load_documents(path, "novel")

    assert documents[0].corpus_name == "Novel-1"
    assert documents[0].character_count == 7
    assert documents[0].word_count == 2


def test_loads_parquet_corpus_rows(tmp_path: Path) -> None:
    path = tmp_path / "novel.parquet"
    pd.DataFrame(
        [{"corpus_name": "Novel-2", "context": "three words here"}]
    ).to_parquet(path)

    documents = GraphRAGBenchLoader().load_documents(path, "novel")

    assert documents[0].corpus_name == "Novel-2"
    assert documents[0].word_count == 3


def test_rejects_empty_context(tmp_path: Path) -> None:
    path = tmp_path / "novel.json"
    path.write_text(
        json.dumps([{"corpus_name": "empty", "context": ""}]),
        encoding="utf-8",
    )

    with pytest.raises(AppError) as error:
        GraphRAGBenchLoader().load_documents(path, "novel")

    assert error.value.code == "INVALID_CORPUS"


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "novel.txt"
    path.write_text("not a supported corpus", encoding="utf-8")

    with pytest.raises(AppError) as error:
        GraphRAGBenchLoader().load_documents(path, "novel")

    assert error.value.code == "UNSUPPORTED_CORPUS"
