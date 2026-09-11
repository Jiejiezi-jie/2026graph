from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.official_backends.base import METHODS


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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("results_api/p1/analysis/silver_labels.jsonl"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(
            "results_api/p1/analysis/router_experiments/no_all_failed_bge_m3/"
            "question_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_api/p1/analysis/router_experiments/bge_m3_5fold_cv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = [row for row in load_jsonl(args.labels) if not row["all_failed"]]
    embedding_data = np.load(args.embeddings)
    embedding_ids = embedding_data["question_ids"].tolist()
    embedding_map = {
        str(question_id): vector
        for question_id, vector in zip(embedding_ids, embedding_data["embeddings"])
    }
    if set(row["question_id"] for row in labels) != set(embedding_map):
        raise ValueError("Labels and embedding IDs are not aligned")

    development = [row for row in labels if row["split"] in ("train", "validation")]
    test = [row for row in labels if row["split"] == "test"]
    x_dev = np.stack([embedding_map[row["question_id"]] for row in development])
    y_dev = np.array([row["silver_label"] for row in development])
    x_test = np.stack([embedding_map[row["question_id"]] for row in test])
    y_test = [row["silver_label"] for row in test]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    candidates = []
    for class_weight in (None, "balanced"):
        for c_value in (0.01, 0.05, 0.1, 0.5, 1.0):
            fold_scores = []
            all_truth = []
            all_predicted = []
            for train_indices, validation_indices in splitter.split(x_dev, y_dev):
                model = LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=2_000,
                    random_state=args.seed,
                )
                model.fit(x_dev[train_indices], y_dev[train_indices])
                predicted = model.predict(x_dev[validation_indices]).tolist()
                truth = y_dev[validation_indices].tolist()
                fold_scores.append(score(truth, predicted))
                all_truth.extend(truth)
                all_predicted.extend(predicted)
            candidates.append(
                {
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
    model = LogisticRegression(
        C=best["c_value"],
        class_weight=best["class_weight"],
        max_iter=2_000,
        random_state=args.seed,
    )
    model.fit(x_dev, y_dev)
    predicted = model.predict(x_test).tolist()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "bge_m3_5fold_cv_no_all_failed",
        "all_failed_policy": "excluded from development and test evaluation",
        "development_counts": dict(Counter(row["split"] for row in development)),
        "test_count": len(test),
        "development_label_distribution": dict(Counter(y_dev.tolist())),
        "test_label_distribution": dict(Counter(y_test)),
        "selected_hyperparameters": {
            key: value for key, value in best.items() if key != "fold_scores"
        },
        "test": score(y_test, predicted),
        "embedding_dimension": int(x_dev.shape[1]),
        "folds": 5,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "tuning.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
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
    joblib.dump(model, output_dir / "router.joblib")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
