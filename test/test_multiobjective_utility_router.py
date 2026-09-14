from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "result/shared_expanded"
    / "p1"
    / "multiobjective_utility_router"
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_multiobjective_selection_uses_development_only() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    assert protocol["total_questions"] == 300
    assert protocol["development_count"] == 276
    assert protocol["frozen_test_count"] == 24
    assert protocol["model_inputs"] == ["question BGE-M3 embedding"]
    assert protocol["test_used_for_training_or_selection"] is False
    assert protocol["new_api_calls"] == 0
    assert summary["selected_parameters"] == {
        "ridge_alpha": 0.001,
        "evidence_weight": 0.15,
    }
    assert len(json.loads((OUTPUT / "candidates.json").read_text(encoding="utf-8"))) == 48


def test_multiobjective_oof_improves_both_development_objectives() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    delta = summary["development_oof"]["delta"]
    assert delta["answer_correctness"] > 0
    assert delta["evidence_recall"] > 0
    rows = load_jsonl(OUTPUT / "oof_predictions.jsonl")
    assert len(rows) == 276
    assert len({row["question_id"] for row in rows}) == 276
    assert {row["predicted_route"] for row in rows}.issubset(
        {"vector", "lightrag", "pathrag"}
    )


def test_multiobjective_frozen_test_improves_both_reference_metrics() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    result = summary["frozen_test"]["multiobjective_utility_router"]
    reference = summary["frozen_test"]["reference_120_weighted_hard_router"]
    assert result["answer_correctness"] > reference["answer_correctness"]
    assert result["evidence_recall"] > reference["evidence_recall"]
    assert result["route_distribution"] == {
        "vector": 7,
        "lightrag": 7,
        "pathrag": 10,
    }
    rows = load_jsonl(OUTPUT / "test_predictions.jsonl")
    assert len(rows) == 24
    assert len({row["question_id"] for row in rows}) == 24
