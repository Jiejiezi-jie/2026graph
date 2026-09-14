from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.backend.common.base import METHODS
from src.backend.common.model_client import TransformersEmbeddingClient
from src.router import build_router


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def metrics(truth: list[str], predicted: list[str]) -> dict[str, Any]:
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
        "prediction_distribution": dict(Counter(predicted)),
    }


def candidate_key(row: dict[str, Any]) -> tuple[float, float]:
    return row["validation"]["macro_f1"], row["validation"]["accuracy"]


def save_experiment(
    output_dir: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    best: dict[str, Any],
    model: Any,
    test_predictions: list[str],
    experiment_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = output_dir / experiment_name
    target.mkdir(parents=True, exist_ok=True)
    test = rows_by_split["test"]
    truth = [row["silver_label"] for row in test]
    summary = {
        "experiment": experiment_name,
        "all_failed_policy": (
            "included in train, validation, and test"
            if "with_all_failed" in experiment_name
            else "excluded from train, validation, and test"
        ),
        "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "label_distribution": {
            split: dict(Counter(row["silver_label"] for row in rows))
            for split, rows in rows_by_split.items()
        },
        "selected_hyperparameters": {
            key: value for key, value in best.items() if key != "validation"
        },
        "validation": best["validation"],
        "test": metrics(truth, test_predictions),
        **(extra or {}),
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "tuning.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (target / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, predicted in zip(test, test_predictions):
            handle.write(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "question": row["question"],
                        "silver_label": row["silver_label"],
                        "predicted_label": predicted,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    joblib.dump(model, target / "router.joblib")
    return summary


def run_tfidf(
    rows_by_split: dict[str, list[dict[str, Any]]], output_dir: Path, seed: int
) -> dict[str, Any]:
    train = rows_by_split["train"]
    validation = rows_by_split["validation"]
    candidates: list[dict[str, Any]] = []
    factories: dict[tuple[str, str | None, float], Callable[[], Any]] = {}
    for feature_kind in ("word", "word_char"):
        for class_weight in (None, "balanced"):
            for c_value in (0.5, 1.0, 2.0):
                key = (feature_kind, class_weight, c_value)
                factories[key] = lambda fk=feature_kind, cw=class_weight, c=c_value: (
                    build_router(fk, cw, c, seed)
                )
                model = factories[key]()
                model.fit(
                    [row["question"] for row in train],
                    [row["silver_label"] for row in train],
                )
                predicted = model.predict(
                    [row["question"] for row in validation]
                ).tolist()
                candidates.append(
                    {
                        "feature_kind": feature_kind,
                        "class_weight": class_weight,
                        "c_value": c_value,
                        "validation": metrics(
                            [row["silver_label"] for row in validation], predicted
                        ),
                    }
                )
    best = max(candidates, key=candidate_key)
    model = factories[
        (best["feature_kind"], best["class_weight"], best["c_value"])
    ]()
    fitted = train + validation
    model.fit(
        [row["question"] for row in fitted],
        [row["silver_label"] for row in fitted],
    )
    predicted = model.predict(
        [row["question"] for row in rows_by_split["test"]]
    ).tolist()
    return save_experiment(
        output_dir,
        rows_by_split,
        candidates,
        best,
        model,
        predicted,
        "no_all_failed_tfidf",
    )


async def embed_questions(
    rows: list[dict[str, Any]], model_path: Path, device: str, batch_size: int
) -> np.ndarray:
    client = TransformersEmbeddingClient(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        max_length=512,
        normalize=True,
    )
    try:
        return await client.embed([row["question"] for row in rows])
    finally:
        client.unload()


def run_bge(
    rows_by_split: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    seed: int,
    model_path: Path,
    device: str,
    batch_size: int,
    include_all_failed: bool,
) -> dict[str, Any]:
    ordered_rows = [
        row
        for split in ("train", "validation", "test")
        for row in rows_by_split[split]
    ]
    embeddings = asyncio.run(
        embed_questions(ordered_rows, model_path, device, batch_size)
    )
    train_n = len(rows_by_split["train"])
    validation_n = len(rows_by_split["validation"])
    train_x = embeddings[:train_n]
    validation_x = embeddings[train_n : train_n + validation_n]
    test_x = embeddings[train_n + validation_n :]
    train_y = [row["silver_label"] for row in rows_by_split["train"]]
    validation_y = [row["silver_label"] for row in rows_by_split["validation"]]
    candidates: list[dict[str, Any]] = []
    for class_weight in (None, "balanced"):
        for c_value in (0.1, 0.5, 1.0, 2.0, 5.0):
            model = LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=2_000,
                random_state=seed,
            )
            model.fit(train_x, train_y)
            predicted = model.predict(validation_x).tolist()
            candidates.append(
                {
                    "feature_kind": "bge_m3_cls",
                    "class_weight": class_weight,
                    "c_value": c_value,
                    "validation": metrics(validation_y, predicted),
                }
            )
    best = max(candidates, key=candidate_key)
    model = LogisticRegression(
        C=best["c_value"],
        class_weight=best["class_weight"],
        max_iter=2_000,
        random_state=seed,
    )
    model.fit(
        np.concatenate([train_x, validation_x]), train_y + validation_y
    )
    predicted = model.predict(test_x).tolist()
    experiment_name = (
        "with_all_failed_bge_m3" if include_all_failed else "no_all_failed_bge_m3"
    )
    target = output_dir / experiment_name
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "question_embeddings.npz",
        question_ids=np.array([row["question_id"] for row in ordered_rows]),
        embeddings=embeddings,
    )
    return save_experiment(
        output_dir,
        rows_by_split,
        candidates,
        best,
        model,
        predicted,
        experiment_name,
        {
            "embedding_model": str(model_path),
            "embedding_device": device,
            "embedding_dimension": int(embeddings.shape[1]),
            "pooling": "first token (CLS)",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("result/api/p1/analysis/silver_labels.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/api/p1/analysis/router_experiments"),
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment", choices=("tfidf", "bge", "all"), default="all"
    )
    parser.add_argument(
        "--include-all-failed",
        action="store_true",
        help="Keep all_failed rows in train, validation, and test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.labels)
    rows_by_split = {
        split: [
            row
            for row in rows
            if row["split"] == split
            and (args.include_all_failed or not row["all_failed"])
        ]
        for split in ("train", "validation", "test")
    }
    all_failed_count = sum(row["all_failed"] for row in rows)
    policy = "included" if args.include_all_failed else "excluded"
    print(f"{policy.capitalize()} {all_failed_count} all_failed rows", flush=True)
    reports = []
    if args.experiment in ("tfidf", "all"):
        reports.append(run_tfidf(rows_by_split, args.output_dir, args.seed))
    if args.experiment in ("bge", "all"):
        if args.model_path is None:
            raise ValueError("--model-path is required for the BGE-M3 experiment")
        reports.append(
            run_bge(
                rows_by_split,
                args.output_dir,
                args.seed,
                args.model_path,
                args.device,
                args.batch_size,
                args.include_all_failed,
            )
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
