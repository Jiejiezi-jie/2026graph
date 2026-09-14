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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router.analyze_router_data_expansion import build_labels, resolve, route_policy, validate_and_merge
from src.router.run_weighted_router_experiment import classify_weight, metrics, sha256
from src.backend.common.base import METHODS
from src.router import build_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 300-question hard-label versus sample-weight grid."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/router_hard_weight_grid_300.json"),
    )
    return parser.parse_args()


def make_model(
    feature: str,
    feature_kind: str,
    class_weight: str | None,
    c_value: float,
    seed: int,
    max_iter: int,
) -> Any:
    if feature == "tfidf":
        return build_router(feature_kind, class_weight, c_value, seed)
    return LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
    )


def fit_model(
    model: Any,
    feature: str,
    questions: np.ndarray,
    vectors: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray | None,
) -> Any:
    x = questions.tolist() if feature == "tfidf" else vectors
    if sample_weights is None:
        return model.fit(x, labels)
    if feature == "tfidf":
        return model.fit(x, labels, classifier__sample_weight=sample_weights)
    return model.fit(x, labels, sample_weight=sample_weights)


def predict_model(
    model: Any,
    feature: str,
    questions: np.ndarray,
    vectors: np.ndarray,
) -> list[str]:
    x = questions.tolist() if feature == "tfidf" else vectors
    return model.predict(x).tolist()


def tune(
    config: dict[str, Any],
    feature: str,
    questions: np.ndarray,
    vectors: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    weighted: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    candidates: list[dict[str, Any]] = []
    candidate_predictions: list[list[str]] = []
    feature_config = config[feature]
    for feature_kind in feature_config["feature_kinds"]:
        for class_weight in config["class_weight_candidates"]:
            for c_value in feature_config["C_candidates"]:
                oof = np.empty(len(labels), dtype=object)
                fold_scores = []
                for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
                    model = make_model(
                        feature,
                        feature_kind,
                        class_weight,
                        float(c_value),
                        int(config["seed"]),
                        int(config["max_iter"]),
                    )
                    fit_model(
                        model,
                        feature,
                        questions[train_index],
                        vectors[train_index],
                        labels[train_index],
                        sample_weights[train_index] if weighted else None,
                    )
                    predicted = predict_model(
                        model,
                        feature,
                        questions[valid_index],
                        vectors[valid_index],
                    )
                    oof[valid_index] = predicted
                    fold_scores.append(
                        {
                            "fold": fold_number,
                            "unweighted": metrics(labels[valid_index].tolist(), predicted),
                            "sample_weighted": metrics(
                                labels[valid_index].tolist(),
                                predicted,
                                sample_weights[valid_index],
                            ),
                        }
                    )
                predictions = oof.tolist()
                candidates.append(
                    {
                        "feature_kind": feature_kind,
                        "class_weight": class_weight,
                        "C": float(c_value),
                        "fold_scores": fold_scores,
                        "cross_validation": metrics(labels.tolist(), predictions),
                        "sample_weighted_cross_validation": metrics(
                            labels.tolist(), predictions, sample_weights
                        ),
                    }
                )
                candidate_predictions.append(predictions)
    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["cross_validation"]["macro_f1"],
            candidates[index]["cross_validation"]["accuracy"],
        ),
    )
    return candidates, candidates[best_index], candidate_predictions[best_index]


def run_one(
    config: dict[str, Any],
    mode: str,
    threshold: float,
    margin: float,
    feature: str,
    development_ids: list[str],
    test_ids: list[str],
    labels_by_id: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
    rows_by_id: dict[str, dict[str, dict[str, Any]]],
    embeddings: dict[str, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> dict[str, Any]:
    weighted = mode == "sample_weighted"
    questions = np.asarray([labels_by_id[qid]["question"] for qid in development_ids], dtype=object)
    vectors = np.stack([embeddings[qid] for qid in development_ids])
    targets = np.asarray([labels_by_id[qid]["silver_label"] for qid in development_ids])
    sample_weights = np.asarray([diagnostics[qid]["weight"] for qid in development_ids])
    candidates, best, oof_predictions = tune(
        config, feature, questions, vectors, targets, sample_weights, folds, weighted
    )

    model = make_model(
        feature,
        best["feature_kind"],
        best["class_weight"],
        float(best["C"]),
        int(config["seed"]),
        int(config["max_iter"]),
    )
    fit_model(
        model,
        feature,
        questions,
        vectors,
        targets,
        sample_weights if weighted else None,
    )
    test_questions = np.asarray([labels_by_id[qid]["question"] for qid in test_ids], dtype=object)
    test_vectors = np.stack([embeddings[qid] for qid in test_ids])
    test_targets = [labels_by_id[qid]["silver_label"] for qid in test_ids]
    test_predictions = predict_model(model, feature, test_questions, test_vectors)

    name = f"threshold_{threshold:.2f}_margin_{margin:.2f}_{feature}"
    target_dir = output_dir / mode / name
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "tuning.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (target_dir / "oof_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for qid, truth, prediction in zip(development_ids, targets.tolist(), oof_predictions):
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "silver_label": truth,
                        "predicted_label": prediction,
                        "weight_category": diagnostics[qid]["category"],
                        "sample_weight": diagnostics[qid]["weight"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (target_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for qid, truth, prediction in zip(test_ids, test_targets, test_predictions):
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "silver_label": truth,
                        "predicted_label": prediction,
                        "weight_category": diagnostics[qid]["category"],
                        "sample_weight": diagnostics[qid]["weight"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    joblib.dump(model, target_dir / "router.joblib")
    result = {
        "mode": mode,
        "correctness_threshold": threshold,
        "best_margin": margin,
        "feature": feature,
        "development_count": len(development_ids),
        "test_count": len(test_ids),
        "development_label_distribution": dict(Counter(targets.tolist())),
        "test_label_distribution": dict(Counter(test_targets)),
        "development_weight_category_distribution": dict(
            Counter(diagnostics[qid]["category"] for qid in development_ids)
        ),
        "cross_validation": best["cross_validation"],
        "sample_weighted_cross_validation": best["sample_weighted_cross_validation"],
        "test": metrics(test_targets, test_predictions),
        "majority_accuracy": max(Counter(test_targets).values()) / len(test_targets),
        "selected_hyperparameters": {
            "feature_kind": best["feature_kind"],
            "class_weight": best["class_weight"],
            "C": best["C"],
        },
        "test_policy": route_policy(test_ids, test_predictions, rows_by_id),
    }
    (target_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


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
        raise AssertionError("Expected 276 development, 24 frozen test, and 300 total questions")

    embedding_path = resolve(config["embedding_cache"])
    embedding_data = np.load(embedding_path)
    embeddings = {
        str(qid): vector
        for qid, vector in zip(
            embedding_data["question_ids"].tolist(), embedding_data["embeddings"]
        )
    }
    if not set(rows_by_id).issubset(embeddings):
        raise ValueError("BGE-M3 embedding cache is incomplete")

    strata = np.asarray([rows_by_id[qid]["vector"]["question_type"] for qid in development_ids])
    folds = list(
        StratifiedKFold(
            n_splits=int(config["folds"]),
            shuffle=True,
            random_state=int(config["seed"]),
        ).split(np.arange(len(development_ids)), strata)
    )
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {"hard": [], "sample_weighted": []}
    for threshold in config["thresholds"]:
        for margin in config["best_margins"]:
            labels_by_id = build_labels(rows_by_id, float(threshold), float(margin))
            diagnostics = {
                qid: classify_weight(
                    method_rows,
                    float(threshold),
                    float(margin),
                    config["weights"],
                )
                for qid, method_rows in rows_by_id.items()
            }
            for feature in config["features"]:
                for mode in ("hard", "sample_weighted"):
                    results[mode].append(
                        run_one(
                            config,
                            mode,
                            float(threshold),
                            float(margin),
                            feature,
                            development_ids,
                            test_ids,
                            labels_by_id,
                            diagnostics,
                            rows_by_id,
                            embeddings,
                            folds,
                            output_dir,
                        )
                    )

    summary = {
        "status": "completed offline; no API calls",
        "protocol": {
            "total_questions": 300,
            "development_count": 276,
            "frozen_test_count": 24,
            "folds": config["folds"],
            "fold_stratification": "question_type",
            "seed": config["seed"],
            "thresholds": config["thresholds"],
            "best_margins": config["best_margins"],
            "features": config["features"],
            "weights": config["weights"],
            "hard_definition": "one hard silver label; every development row has weight 1.0",
            "sample_weighted_definition": "same hard silver labels; stable=1.0, ambiguous=0.5, all_failed=0.3",
            "hyperparameter_selection": "OOF Macro-F1, then OOF accuracy",
            "evaluation_protocol": protocol,
            "scikit_learn_version": sklearn.__version__,
            "new_api_calls": 0,
            "input_sha256": input_hashes
            | {
                "embedding_cache": sha256(embedding_path),
                "expansion_config": sha256(expansion_config_path),
                "config": sha256(config_path),
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
