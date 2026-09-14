import json
from pathlib import Path

from app.adapters.graphrag_bench import GraphRAGBenchLoader
from app.domain.models import CorpusDocument
from app.services.dataset_service import DatasetService, corpus_fingerprint


def test_selects_shortest_document_and_persists_it(tmp_path: Path) -> None:
    path = tmp_path / "novel.json"
    path.write_text(
        json.dumps(
            [
                {"corpus_name": "long", "context": "one two three"},
                {"corpus_name": "short", "context": "one two"},
            ]
        ),
        encoding="utf-8",
    )
    service = DatasetService(GraphRAGBenchLoader(), tmp_path / "active")

    selected = service.select_shortest(path, "novel")

    assert selected.corpus_name == "short"
    assert service.load_active() == selected
    manifest = json.loads((tmp_path / "active" / "manifest.json").read_text())
    assert manifest["fingerprint"] == corpus_fingerprint(selected)
    assert manifest["source_path"] == str(path.resolve())


def test_equal_length_documents_are_selected_by_name(tmp_path: Path) -> None:
    path = tmp_path / "novel.json"
    path.write_text(
        json.dumps(
            [
                {"corpus_name": "zeta", "context": "same"},
                {"corpus_name": "alpha", "context": "size"},
            ]
        ),
        encoding="utf-8",
    )
    service = DatasetService(GraphRAGBenchLoader(), tmp_path / "active")

    assert service.select_shortest(path, "novel").corpus_name == "alpha"


def test_load_active_returns_none_before_selection(tmp_path: Path) -> None:
    service = DatasetService(GraphRAGBenchLoader(), tmp_path / "active")

    assert service.load_active() is None


def test_fingerprint_covers_subset_name_and_context() -> None:
    base = CorpusDocument(
        corpus_name="Novel-1",
        context="same text",
        subset="novel",
        character_count=9,
        word_count=2,
    )
    changed = base.model_copy(update={"context": "other text"})

    assert corpus_fingerprint(base) != corpus_fingerprint(changed)
