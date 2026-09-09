from app.domain.models import QueryMode, QueryResult
from scripts.smoke_lightrag import build_summary


def test_summary_contains_action_and_counts() -> None:
    result = QueryResult(
        query="Who is Alice?",
        mode=QueryMode.MIX,
        answer="Alice is the protagonist.",
        latency_ms=12.5,
        context_text="context",
        entities=[{"entity_name": "ALICE"}],
        relationships=[],
        chunks=[{"content": "Alice appears."}],
        references=[],
    )

    summary = build_summary("Novel-test", "built", result)

    assert summary["status"] == "success"
    assert summary["index_action"] == "built"
    assert summary["counts"] == {
        "entities": 1,
        "relationships": 0,
        "chunks": 1,
        "references": 0,
    }
