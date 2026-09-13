from __future__ import annotations

import numpy as np

from scripts.run_weighted_router_experiment import (
    classify_weight,
    estimator,
    fit_estimator,
    metrics,
)


WEIGHTS = {"stable": 1.0, "ambiguous": 0.5, "all_failed": 0.3}


def method_rows(vector: float, lightrag: float, pathrag: float) -> dict:
    return {
        "vector": {"answer_correctness": vector},
        "lightrag": {"answer_correctness": lightrag},
        "pathrag": {"answer_correctness": pathrag},
    }


def test_weight_categories_follow_frozen_eligibility_rule() -> None:
    stable = classify_weight(method_rows(0.80, 0.65, 0.50), 0.60, 0.10, WEIGHTS)
    ambiguous = classify_weight(
        method_rows(0.80, 0.75, 0.61), 0.60, 0.10, WEIGHTS
    )
    failed = classify_weight(method_rows(0.59, 0.58, 0.40), 0.60, 0.10, WEIGHTS)

    assert (stable["category"], stable["weight"]) == ("stable", 1.0)
    assert stable["eligible_methods"] == ["vector"]
    assert (ambiguous["category"], ambiguous["weight"]) == ("ambiguous", 0.5)
    assert ambiguous["eligible_methods"] == ["vector", "lightrag"]
    assert (failed["category"], failed["weight"]) == ("all_failed", 0.3)
    assert failed["eligible_methods"] == []


def test_logistic_regression_accepts_frozen_sample_weights() -> None:
    config = {
        "seed": 42,
        "logistic_regression": {"max_iter": 2000},
    }
    features = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5], [0.4, 0.6]]
    )
    labels = np.array(
        ["vector", "vector", "lightrag", "lightrag", "pathrag", "pathrag"]
    )
    weights = np.array([1.0, 1.0, 0.5, 0.5, 0.3, 0.3])
    model = fit_estimator(
        estimator(config, c_value=1.0, class_weight="balanced"),
        features,
        labels,
        weights,
    )

    probabilities = model.predict_proba(features)
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_weighted_metrics_do_not_change_the_unweighted_count() -> None:
    result = metrics(
        ["vector", "lightrag", "pathrag"],
        ["vector", "vector", "pathrag"],
        [1.0, 0.5, 0.3],
    )
    assert result["n"] == 3
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
