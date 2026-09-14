from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router.analyze_router_data_expansion import resolve, validate_and_merge
from src.router.run_multiobjective_utility_router import full_policy_metrics, sha256
from src.backend.common.base import METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare balanced-utility labels and direct utility regression."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/router_balanced_utility_labels_flash.json"),
    )
    return parser.parse_args()


def method_targets(
    ids: list[str], rows_by_id: dict[str, dict[str, dict[str, Any]]]
) -> np.ndarray:
    return np.asarray(
        [
            [
                [
                    float(rows_by_id[qid][method]["answer_correctness"]),
                    float(rows_by_id[qid][method]["evidence_recall"]),
                ]
                for method in METHODS
            ]
            for qid in ids
        ],
        dtype=float,
    )


def utility_scores(kind: str, targets: np.ndarray) -> np.ndarray:
    correctness = targets[:, :, 0]
    evidence = targets[:, :, 1]
    if kind == "harmonic":
        result = np.zeros_like(correctness)
        np.divide(
            2 * correctness * evidence,
            correctness + evidence,
            out=result,
            where=(correctness + evidence) > 0,
        )
        return result
    if kind == "geometric":
        return np.sqrt(correctness * evidence)
    if kind == "minimum":
        return np.minimum(correctness, evidence)
    if kind == "mean":
        return (correctness + evidence) / 2
    if kind == "ac75_er25":
        return 0.75 * correctness + 0.25 * evidence
    if kind == "ac60_er40":
        return 0.60 * correctness + 0.40 * evidence
    raise ValueError(f"Unsupported utility: {kind}")


def cheapest_within_tolerance(scores: np.ndarray, tolerance: float) -> np.ndarray:
    """Choose the lowest-cost method whose score is within tolerance of the best."""

    maximum = np.max(scores, axis=1, keepdims=True)
    return np.argmax(scores >= maximum - tolerance, axis=1)


def policy_metrics(routes: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    selected = targets[np.arange(len(routes)), routes]
    correctness = float(np.mean(selected[:, 0]))
    evidence = float(np.mean(selected[:, 1]))
    harmonic = (
        2 * correctness * evidence / (correctness + evidence)
        if correctness + evidence > 0
        else 0.0
    )
    return {
        "n": len(routes),
        "answer_correctness": correctness,
        "evidence_recall": evidence,
        "joint_harmonic_mean": harmonic,
        "route_distribution": {
            method: int(np.sum(routes == index))
            for index, method in enumerate(METHODS)
        },
    }


def gap_weights(
    utilities: np.ndarray, minimum_weight: float, gap_scale: float
) -> np.ndarray:
    ordered = np.sort(utilities, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    confidence = np.clip(gap / gap_scale, 0.0, 1.0)
    return minimum_weight + (1.0 - minimum_weight) * confidence


def candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    scores = candidate["oof_policy"]
    # The final fields make exact ties deterministic without consuming test scores.
    return (
        scores["joint_harmonic_mean"],
        scores["answer_correctness"],
        scores["evidence_recall"],
        candidate["family"] == "logistic_utility_labels",
        -float(candidate["parameter"]),
        -float(candidate["tie_tolerance"]),
        candidate["utility"],
    )


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expansion_config_path = resolve(config["expansion_config"])
    expansion_config = json.loads(expansion_config_path.read_text(encoding="utf-8"))
    rows_by_id, baseline_ids, expansion_ids, protocol, input_hashes = validate_and_merge(
        expansion_config
    )

    test_ids = sorted(
        qid for qid in baseline_ids if rows_by_id[qid]["vector"]["split"] == "test"
    )
    development_ids = sorted((baseline_ids - set(test_ids)) | expansion_ids)
    if (len(development_ids), len(test_ids), len(rows_by_id)) != (276, 24, 300):
        raise AssertionError("Expected 276 development, 24 frozen test, 300 total")

    embedding_path = resolve(config["embedding_cache"])
    embedding_data = np.load(embedding_path)
    embedding_map = {
        str(qid): vector
        for qid, vector in zip(
            embedding_data["question_ids"].tolist(), embedding_data["embeddings"]
        )
    }
    if not set(rows_by_id).issubset(embedding_map):
        raise ValueError("Question embedding cache is incomplete")

    development_x = np.stack([embedding_map[qid] for qid in development_ids])
    development_targets = method_targets(development_ids, rows_by_id)
    strata = np.asarray(
        [rows_by_id[qid]["vector"]["question_type"] for qid in development_ids]
    )
    folds = list(
        StratifiedKFold(
            n_splits=int(config["folds"]),
            shuffle=True,
            random_state=int(config["seed"]),
        ).split(development_x, strata)
    )

    candidates: list[dict[str, Any]] = []
    prediction_cache: dict[int, np.ndarray] = {}
    for utility in config["utility_candidates"]:
        utilities = utility_scores(utility, development_targets)
        for alpha_value in config["ridge_alpha_candidates"]:
            alpha = float(alpha_value)
            oof_scores = np.zeros_like(utilities)
            for train_index, validation_index in folds:
                model = Ridge(alpha=alpha)
                model.fit(development_x[train_index], utilities[train_index])
                oof_scores[validation_index] = model.predict(
                    development_x[validation_index]
                )
            for tolerance_value in config["tie_tolerances"]:
                tolerance = float(tolerance_value)
                routes = cheapest_within_tolerance(oof_scores, tolerance)
                candidate = {
                    "family": "ridge_direct_utility",
                    "utility": utility,
                    "parameter": alpha,
                    "tie_tolerance": tolerance,
                    "class_weight": None,
                    "sample_weight": "none",
                    "oof_policy": policy_metrics(routes, development_targets),
                }
                candidates.append(candidate)
                prediction_cache[id(candidate)] = routes

        for tolerance_value in config["tie_tolerances"]:
            tolerance = float(tolerance_value)
            labels = cheapest_within_tolerance(utilities, tolerance)
            utility_gap_weights = gap_weights(
                utilities,
                float(config["minimum_sample_weight"]),
                float(config["utility_gap_scale"]),
            )
            for class_weight in config["logistic_class_weight_candidates"]:
                for sample_weight_name in config["sample_weight_candidates"]:
                    weights = (
                        np.ones(len(labels), dtype=float)
                        if sample_weight_name == "none"
                        else utility_gap_weights
                    )
                    for c_value in config["logistic_c_candidates"]:
                        c_value = float(c_value)
                        routes = np.zeros(len(labels), dtype=int)
                        for train_index, validation_index in folds:
                            model = LogisticRegression(
                                C=c_value,
                                class_weight=class_weight,
                                max_iter=int(config["max_iter"]),
                                random_state=int(config["seed"]),
                            )
                            model.fit(
                                development_x[train_index],
                                labels[train_index],
                                sample_weight=weights[train_index],
                            )
                            routes[validation_index] = model.predict(
                                development_x[validation_index]
                            )
                        candidate = {
                            "family": "logistic_utility_labels",
                            "utility": utility,
                            "parameter": c_value,
                            "tie_tolerance": tolerance,
                            "class_weight": class_weight,
                            "sample_weight": sample_weight_name,
                            "oof_policy": policy_metrics(routes, development_targets),
                            "development_label_distribution": {
                                method: int(np.sum(labels == index))
                                for index, method in enumerate(METHODS)
                            },
                        }
                        candidates.append(candidate)
                        prediction_cache[id(candidate)] = routes

    # Selection is complete using development OOF outcomes. Test is first loaded below.
    selected = max(candidates, key=candidate_key)
    oof_routes = prediction_cache[id(selected)]
    selected_utility = utility_scores(selected["utility"], development_targets)
    if selected["family"] == "ridge_direct_utility":
        final_model: Any = Ridge(alpha=float(selected["parameter"]))
        final_model.fit(development_x, selected_utility)
    else:
        labels = cheapest_within_tolerance(
            selected_utility, float(selected["tie_tolerance"])
        )
        weights = (
            np.ones(len(labels), dtype=float)
            if selected["sample_weight"] == "none"
            else gap_weights(
                selected_utility,
                float(config["minimum_sample_weight"]),
                float(config["utility_gap_scale"]),
            )
        )
        final_model = LogisticRegression(
            C=float(selected["parameter"]),
            class_weight=selected["class_weight"],
            max_iter=int(config["max_iter"]),
            random_state=int(config["seed"]),
        )
        final_model.fit(development_x, labels, sample_weight=weights)

    test_x = np.stack([embedding_map[qid] for qid in test_ids])
    test_targets = method_targets(test_ids, rows_by_id)
    if selected["family"] == "ridge_direct_utility":
        test_routes = cheapest_within_tolerance(
            final_model.predict(test_x), float(selected["tie_tolerance"])
        )
    else:
        test_routes = final_model.predict(test_x)

    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "router.joblib"
    joblib.dump(final_model, model_path)
    with (output_dir / "oof_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for qid, route in zip(development_ids, oof_routes):
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "split": rows_by_id[qid]["vector"]["split"],
                        "predicted_route": METHODS[int(route)],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for qid, route in zip(test_ids, test_routes):
            method = METHODS[int(route)]
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": rows_by_id[qid]["vector"]["question"],
                        "predicted_route": method,
                        "actual_selected_answer_correctness": float(
                            rows_by_id[qid][method]["answer_correctness"]
                        ),
                        "actual_selected_evidence_recall": float(
                            rows_by_id[qid][method]["evidence_recall"]
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    candidates_path = output_dir / "candidates.json"
    candidates_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    test_policy = full_policy_metrics(test_ids, test_routes, rows_by_id)
    summary = {
        "status": "completed offline; no API, retrieval, generation, or evaluation calls",
        "protocol": {
            "total_questions": 300,
            "development_count": 276,
            "frozen_test_count": 24,
            "feature": config["feature"],
            "folds": config["folds"],
            "fold_stratification": "question_type",
            "seed": config["seed"],
            "candidate_count": len(candidates),
            "selection_metric": config["selection_metric"],
            "test_used_for_training_or_selection": False,
            "evaluation_protocol": protocol,
            "scikit_learn_version": sklearn.__version__,
            "new_api_calls": 0,
            "input_sha256": input_hashes
            | {
                "embedding_cache": sha256(embedding_path),
                "config": sha256(config_path),
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "selected_candidate": selected,
        "development_oof": selected["oof_policy"],
        "frozen_test": test_policy,
        "artifacts": {
            "model": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256(model_path),
            "oof_predictions": str(
                (output_dir / "oof_predictions.jsonl").relative_to(ROOT)
            ),
            "test_predictions": str(
                (output_dir / "test_predictions.jsonl").relative_to(ROOT)
            ),
            "candidates": str(candidates_path.relative_to(ROOT)),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
