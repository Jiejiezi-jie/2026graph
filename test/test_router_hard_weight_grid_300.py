from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "results_shared_lightrag_expanded"
    / "p1"
    / "router_hard_weight_grid_300"
    / "summary.json"
)


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def keyed(rows: list[dict]) -> dict[tuple[float, float, str], dict]:
    return {
        (row["correctness_threshold"], row["best_margin"], row["feature"]): row
        for row in rows
    }


def test_grid_has_two_complete_twelve_row_modes_and_no_api_calls() -> None:
    summary = load_summary()
    protocol = summary["protocol"]
    assert protocol["total_questions"] == 300
    assert protocol["development_count"] == 276
    assert protocol["frozen_test_count"] == 24
    assert protocol["new_api_calls"] == 0
    assert protocol["weights"] == {
        "stable": 1.0,
        "ambiguous": 0.5,
        "all_failed": 0.3,
    }
    expected = {
        (threshold, margin, feature)
        for threshold in (0.5, 0.6)
        for margin in (0.05, 0.1, 0.15)
        for feature in ("tfidf", "bge_m3")
    }
    for mode in ("hard", "sample_weighted"):
        rows = summary["results"][mode]
        assert len(rows) == 12
        assert set(keyed(rows)) == expected
        assert all(row["development_count"] == 276 for row in rows)
        assert all(row["test_count"] == 24 for row in rows)


def test_weighting_changes_only_training_weight_not_silver_labels() -> None:
    summary = load_summary()
    hard = keyed(summary["results"]["hard"])
    weighted = keyed(summary["results"]["sample_weighted"])
    for key in hard:
        assert hard[key]["development_label_distribution"] == weighted[key][
            "development_label_distribution"
        ]
        assert hard[key]["test_label_distribution"] == weighted[key][
            "test_label_distribution"
        ]


def test_grid_reproduces_existing_060_010_results() -> None:
    summary = load_summary()
    hard = keyed(summary["results"]["hard"])[(0.6, 0.1, "bge_m3")]
    weighted = keyed(summary["results"]["sample_weighted"])[(0.6, 0.1, "bge_m3")]
    expansion = json.loads(
        (
            ROOT
            / "results_shared_lightrag_expanded"
            / "p1"
            / "router_data_expansion"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )["experiments"]["expanded_bge_m3"]
    weighted_expansion = json.loads(
        (
            ROOT
            / "results_shared_lightrag_expanded"
            / "p1"
            / "weighted_router_data_expansion"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )["experiments"]["expanded_weighted"]
    assert hard["cross_validation"] == expansion["cross_validation"]
    assert hard["test"] == expansion["test"]
    assert weighted["cross_validation"] == weighted_expansion["cross_validation"]
    assert weighted["test"] == weighted_expansion["test"]
