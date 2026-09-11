from __future__ import annotations

import json
import subprocess
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from src.official_backends.base import METHODS
from src.router import build_router


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results_api/p1/analysis/router_experiments/rule_feature_5fold_comparison"
CURRENT_LABELS = ROOT / "results_api/p1/analysis/silver_labels.jsonl"
EMBEDDINGS = (
    ROOT
    / "results_api/p1/analysis/router_experiments/with_all_failed_bge_m3/"
    "question_embeddings.npz"
)


def parse_jsonl(content: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in content.splitlines()]


def load_labels() -> dict[str, list[dict[str, Any]]]:
    legacy = subprocess.check_output(
        ["git", "show", "HEAD:results_api/p1/analysis/silver_labels.jsonl"],
        cwd=ROOT,
        text=True,
    )
    return {
        "old_rule": parse_jsonl(legacy),
        "new_rule": parse_jsonl(CURRENT_LABELS.read_text(encoding="utf-8")),
    }


def score(truth: list[str], predicted: list[str]) -> dict[str, Any]:
    return {
        "n": len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(METHODS),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=list(METHODS)
        ).tolist(),
    }


def evaluate_candidates(
    feature: str,
    questions: np.ndarray,
    vectors: np.ndarray,
    targets: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    feature_kinds = ("word", "word_char") if feature == "tfidf" else ("bge_m3_cls",)
    c_values = (0.5, 1.0, 2.0) if feature == "tfidf" else (0.01, 0.05, 0.1, 0.5, 1.0)
    for feature_kind in feature_kinds:
        for class_weight in (None, "balanced"):
            for c_value in c_values:
                fold_scores = []
                all_truth: list[str] = []
                all_predicted: list[str] = []
                for train_indices, validation_indices in folds:
                    if feature == "tfidf":
                        model = build_router(feature_kind, class_weight, c_value, seed)
                        model.fit(
                            questions[train_indices].tolist(), targets[train_indices].tolist()
                        )
                        predicted = model.predict(
                            questions[validation_indices].tolist()
                        ).tolist()
                    else:
                        model = LogisticRegression(
                            C=c_value,
                            class_weight=class_weight,
                            max_iter=2_000,
                            random_state=seed,
                        )
                        model.fit(vectors[train_indices], targets[train_indices])
                        predicted = model.predict(vectors[validation_indices]).tolist()
                    truth = targets[validation_indices].tolist()
                    fold_scores.append(score(truth, predicted))
                    all_truth.extend(truth)
                    all_predicted.extend(predicted)
                candidates.append(
                    {
                        "feature_kind": feature_kind,
                        "class_weight": class_weight,
                        "c_value": c_value,
                        "fold_scores": fold_scores,
                        "cross_validation": score(all_truth, all_predicted),
                    }
                )
    best = max(
        candidates,
        key=lambda row: (
            row["cross_validation"]["macro_f1"],
            row["cross_validation"]["accuracy"],
        ),
    )
    return candidates, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-labels", type=Path, default=CURRENT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS)
    parser.add_argument("--current-margin", type=float, default=0.05)
    args = parser.parse_args()
    seed = 42
    labels_by_rule = load_labels()
    labels_by_rule["new_rule"] = parse_jsonl(
        args.current_labels.read_text(encoding="utf-8")
    )
    old_by_id = {row["question_id"]: row for row in labels_by_rule["old_rule"]}
    new_by_id = {row["question_id"]: row for row in labels_by_rule["new_rule"]}
    if set(old_by_id) != set(new_by_id):
        raise ValueError("Old and new label sets do not contain the same questions")

    development_ids = [
        row["question_id"]
        for row in labels_by_rule["old_rule"]
        if row["split"] in ("train", "validation")
    ]
    test_ids = [
        row["question_id"]
        for row in labels_by_rule["old_rule"]
        if row["split"] == "test"
    ]
    questions = np.array([old_by_id[qid]["question"] for qid in development_ids], dtype=object)
    test_questions = [old_by_id[qid]["question"] for qid in test_ids]

    embedding_data = np.load(args.embeddings)
    embedding_map = {
        str(qid): vector
        for qid, vector in zip(
            embedding_data["question_ids"].tolist(), embedding_data["embeddings"]
        )
    }
    if not set(development_ids + test_ids).issubset(embedding_map):
        raise ValueError("Full BGE-M3 embedding cache is missing question IDs")
    vectors = np.stack([embedding_map[qid] for qid in development_ids])
    test_vectors = np.stack([embedding_map[qid] for qid in test_ids])

    strata = np.array([old_by_id[qid]["question_type"] for qid in development_ids])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(questions, strata))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    rule_descriptions = {
        "old_rule": "correctness >= 0.50; cheapest eligible method",
        "new_rule": (
            "correctness >= 0.60 and >= question best - "
            f"{args.current_margin:.2f}; cheapest eligible method"
        ),
    }
    for rule_name, rows in labels_by_rule.items():
        by_id = {row["question_id"]: row for row in rows}
        targets = np.array([by_id[qid]["silver_label"] for qid in development_ids])
        test_targets = [by_id[qid]["silver_label"] for qid in test_ids]
        for feature in ("tfidf", "bge_m3"):
            candidates, best = evaluate_candidates(
                feature, questions, vectors, targets, folds, seed
            )
            if feature == "tfidf":
                model = build_router(
                    best["feature_kind"], best["class_weight"], best["c_value"], seed
                )
                model.fit(questions.tolist(), targets.tolist())
                predicted = model.predict(test_questions).tolist()
            else:
                model = LogisticRegression(
                    C=best["c_value"],
                    class_weight=best["class_weight"],
                    max_iter=2_000,
                    random_state=seed,
                )
                model.fit(vectors, targets)
                predicted = model.predict(test_vectors).tolist()

            experiment = f"{rule_name}_{feature}"
            target_dir = args.output_dir / experiment
            target_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "experiment": experiment,
                "label_rule": rule_descriptions[rule_name],
                "all_failed_policy": "included as the rule's fallback method label",
                "development_count": len(development_ids),
                "test_count": len(test_ids),
                "development_label_distribution": dict(Counter(targets.tolist())),
                "test_label_distribution": dict(Counter(test_targets)),
                "all_failed": sum(row["all_failed"] for row in rows),
                "selected_hyperparameters": {
                    key: value for key, value in best.items() if key != "fold_scores"
                },
                "test": score(test_targets, predicted),
            }
            reports[experiment] = report
            (target_dir / "summary.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (target_dir / "tuning.json").write_text(
                json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            joblib.dump(model, target_dir / "router.joblib")

    combined = {
        "protocol": {
            "folds": 5,
            "fold_stratification": "question_type",
            "same_fold_membership_for_all_four experiments": True,
            "hyperparameter_selection": "cross-validation Macro-F1, then accuracy",
            "development_count": len(development_ids),
            "test_count": len(test_ids),
        },
        "experiments": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
