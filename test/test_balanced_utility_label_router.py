from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "result/shared_unified_latency"
    / "p1"
    / "balanced_utility_label_router"
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_balanced_utility_selection_uses_development_only() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    assert protocol["total_questions"] == 300
    assert protocol["development_count"] == 276
    assert protocol["frozen_test_count"] == 24
    assert protocol["test_used_for_training_or_selection"] is False
    assert protocol["new_api_calls"] == 0

    candidates = json.loads((OUTPUT / "candidates.json").read_text(encoding="utf-8"))
    assert len(candidates) == protocol["candidate_count"] == 540
    best_harmonic = max(
        candidate["oof_policy"]["joint_harmonic_mean"] for candidate in candidates
    )
    assert summary["selected_candidate"]["oof_policy"]["joint_harmonic_mean"] == best_harmonic


def test_balanced_utility_predictions_are_complete() -> None:
    development = load_jsonl(OUTPUT / "oof_predictions.jsonl")
    test = load_jsonl(OUTPUT / "test_predictions.jsonl")
    assert len(development) == len({row["question_id"] for row in development}) == 276
    assert len(test) == len({row["question_id"] for row in test}) == 24
    allowed = {"vector", "lightrag", "pathrag"}
    assert {row["predicted_route"] for row in development}.issubset(allowed)
    assert {row["predicted_route"] for row in test}.issubset(allowed)
