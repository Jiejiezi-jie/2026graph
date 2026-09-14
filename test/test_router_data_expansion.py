from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.prepare_router_data_expansion import prepare
from src.official_backends.base import METHODS
from src.official_evaluation import valid_evaluation


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_expansion_is_disjoint_and_keeps_test_frozen(tmp_path: Path) -> None:
    config = json.loads(
        (ROOT / "configs" / "router_data_expansion.json").read_text(encoding="utf-8")
    )
    config["question_file"] = str(tmp_path / "expansion.jsonl")
    first, manifest = prepare(config)
    first_bytes = (tmp_path / "expansion.jsonl").read_bytes()
    second, second_manifest = prepare(config)
    assert first == second
    assert manifest == second_manifest
    assert first_bytes == (tmp_path / "expansion.jsonl").read_bytes()

    baseline = load_jsonl(ROOT / config["baseline_inputs"]["vector"])
    baseline_ids = {row["question_id"] for row in baseline}
    assert len(first) == 180
    assert not baseline_ids.intersection(row["id"] for row in first)
    assert Counter(row["split"] for row in first) == {"train": 144, "validation": 36}
    assert Counter(row["question_type"] for row in first) == {
        "Fact Retrieval": 90,
        "Complex Reasoning": 90,
    }
    assert len(manifest["baseline"]["test_ids"]) == 24
    assert manifest["combined"] == {
        "count": 300,
        "development_count": 276,
        "test_count": 24,
    }


def test_completed_expansion_is_aligned_with_baseline_protocol() -> None:
    result_dir = ROOT / "results_shared_lightrag_expanded" / "p1"
    rows = {
        method: load_jsonl(result_dir / f"{method}_evaluated.jsonl")
        for method in METHODS
    }
    assert {method: len(method_rows) for method, method_rows in rows.items()} == {
        method: 180 for method in METHODS
    }
    ids = {
        method: {row["question_id"] for row in method_rows}
        for method, method_rows in rows.items()
    }
    assert len({frozenset(method_ids) for method_ids in ids.values()}) == 1
    assert all(
        valid_evaluation(row) for method_rows in rows.values() for row in method_rows
    )
    protocols = {
        row["evaluation"]["protocol"]
        for method_rows in rows.values()
        for row in method_rows
    }
    baseline_protocol = load_jsonl(
        ROOT / "results_api" / "p1" / "vector_evaluated.jsonl"
    )[0]["evaluation"]["protocol"]
    assert protocols == {baseline_protocol}
    assert all(row["split"] != "test" for row in rows["vector"])


def test_analysis_reproduces_baseline_and_keeps_frozen_test() -> None:
    path = (
        ROOT
        / "results_shared_lightrag_expanded"
        / "p1"
        / "router_data_expansion"
        / "summary.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary["configuration"]
    assert config["baseline_development_count"] == 96
    assert config["expanded_development_count"] == 276
    assert config["frozen_test_count"] == 24
    baseline = summary["experiments"]["baseline_bge_m3"]
    assert baseline["cross_validation"]["accuracy"] == 0.5
    assert baseline["cross_validation"]["macro_f1"] == 0.4164045546534026
    assert baseline["test"]["accuracy"] == 0.4583333333333333
    assert baseline["test"]["macro_f1"] == 0.34848484848484845


def test_weighted_expansion_reuses_prior_scheme_and_frozen_test() -> None:
    path = (
        ROOT
        / "results_shared_lightrag_expanded"
        / "p1"
        / "weighted_router_data_expansion"
        / "summary.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["protocol"]["weights"] == {
        "stable": 1.0,
        "ambiguous": 0.5,
        "all_failed": 0.3,
    }
    assert summary["protocol"]["new_api_calls"] == 0
    assert summary["protocol"]["frozen_test_count"] == 24
    assert summary["baseline_reproduces_prior_weighted_experiment"] is True

    baseline = summary["experiments"]["baseline_weighted"]
    expanded = summary["experiments"]["expanded_weighted"]
    assert baseline["development_count"] == 96
    assert expanded["development_count"] == 276
    assert baseline["cross_validation"]["macro_f1"] == 0.34696257949062087
    assert expanded["cross_validation"]["macro_f1"] == 0.37197688645082133
    assert summary["delta_expanded_minus_baseline_weighted"]["oof_macro_f1"] > 0
    assert (
        summary["delta_expanded_weighted_minus_expanded_unweighted"]["oof_macro_f1"]
        > 0
    )
