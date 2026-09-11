from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from src.official_backends.base import METHODS
from src.router import build_router


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def score(truth: list[str], predicted: list[str]) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("results_api/p1/analysis/silver_labels.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results_api/p1/analysis/router_experiments/tfidf_5fold_cv_no_all_failed"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = [row for row in load_jsonl(args.labels) if not row["all_failed"]]
    development = [row for row in labels if row["split"] in ("train", "validation")]
    test = [row for row in labels if row["split"] == "test"]
    questions = np.array([row["question"] for row in development], dtype=object)
    targets = np.array([row["silver_label"] for row in development])
    test_questions = [row["question"] for row in test]
    test_targets = [row["silver_label"] for row in test]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    candidates = []
    for feature_kind in ("word", "word_char"):
        for class_weight in (None, "balanced"):
            for c_value in (0.5, 1.0, 2.0):
                fold_scores = []
                all_truth = []
                all_predicted = []
                for train_indices, validation_indices in splitter.split(questions, targets):
                    model = build_router(feature_kind, class_weight, c_value, args.seed)
                    model.fit(questions[train_indices].tolist(), targets[train_indices].tolist())
                    predicted = model.predict(questions[validation_indices].tolist()).tolist()
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
    model = build_router(
        best["feature_kind"], best["class_weight"], best["c_value"], args.seed
    )
    model.fit(questions.tolist(), targets.tolist())
    predicted = model.predict(test_questions).tolist()

    summary = {
        "experiment": "tfidf_5fold_cv_no_all_failed",
        "all_failed_policy": "excluded from development and test evaluation",
        "development_counts": dict(Counter(row["split"] for row in development)),
        "test_count": len(test),
        "development_label_distribution": dict(Counter(targets.tolist())),
        "test_label_distribution": dict(Counter(test_targets)),
        "selected_hyperparameters": {
            key: value for key, value in best.items() if key != "fold_scores"
        },
        "test": score(test_targets, predicted),
        "folds": 5,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "tuning.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, predicted_label in zip(test, predicted):
            handle.write(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "silver_label": row["silver_label"],
                        "predicted_label": predicted_label,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    joblib.dump(model, args.output_dir / "router.joblib")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
