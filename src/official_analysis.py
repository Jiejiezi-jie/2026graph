from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.official_backends.base import METHODS
from src.router import build_router


def choose_official_silver(
    method_rows: dict[str, dict[str, Any]], correctness_threshold: float = 0.5
) -> tuple[str, bool]:
    correct = [
        method
        for method in METHODS
        if method_rows[method]["answer_correctness"] >= correctness_threshold
    ]
    if correct:
        chosen = min(
            correct,
            key=lambda method: (
                method_rows[method]["input_tokens"]
                + method_rows[method]["output_tokens"],
                method_rows[method]["total_time_ms"],
                METHODS.index(method),
            ),
        )
        return chosen, False
    chosen = max(
        METHODS,
        key=lambda method: (
            method_rows[method]["answer_correctness"],
            method_rows[method]["rouge_l"],
            method_rows[method]["evidence_recall"],
            -(method_rows[method]["input_tokens"] + method_rows[method]["output_tokens"]),
        ),
    )
    return chosen, True


def _tune_router(train: list[dict], validation: list[dict], seed: int) -> tuple[Any, dict]:
    candidates = []
    for include_failed in (False, True):
        used_train = [row for row in train if include_failed or not row["all_failed"]]
        for feature_kind in ("word", "word_char"):
            for class_weight in (None, "balanced"):
                for c_value in (0.5, 1.0, 2.0):
                    model = build_router(feature_kind, class_weight, c_value, seed)
                    model.fit(
                        [row["question"] for row in used_train],
                        [row["silver_label"] for row in used_train],
                    )
                    predicted = model.predict([row["question"] for row in validation])
                    candidates.append(
                        {
                            "include_all_failed": include_failed,
                            "feature_kind": feature_kind,
                            "class_weight": class_weight,
                            "c_value": c_value,
                            "validation_accuracy": float(
                                accuracy_score(
                                    [row["silver_label"] for row in validation], predicted
                                )
                            ),
                            "validation_macro_f1": float(
                                f1_score(
                                    [row["silver_label"] for row in validation],
                                    predicted,
                                    labels=list(METHODS),
                                    average="macro",
                                    zero_division=0,
                                )
                            ),
                        }
                    )
    best = max(
        candidates,
        key=lambda row: (row["validation_macro_f1"], row["validation_accuracy"]),
    )
    return candidates, best


def _policy(
    ids: list[str], choices: list[str], backend_rows: dict[str, dict[str, dict]]
) -> dict[str, Any]:
    selected = [backend_rows[qid][method] for qid, method in zip(ids, choices)]
    return {
        "n": len(selected),
        "answer_correctness": float(np.mean([row["answer_correctness"] for row in selected])),
        "rouge_l": float(np.mean([row["rouge_l"] for row in selected])),
        "evidence_recall": float(np.mean([row["evidence_recall"] for row in selected])),
        "input_tokens": float(np.mean([row["input_tokens"] for row in selected])),
        "output_tokens": float(np.mean([row["output_tokens"] for row in selected])),
        "latency_ms": float(np.mean([row["total_time_ms"] for row in selected])),
        "route_distribution": dict(Counter(choices)),
    }


def analyze_official(
    rows_by_method: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_rows: dict[str, dict[str, dict]] = {}
    question_info: dict[str, dict] = {}
    for method, rows in rows_by_method.items():
        for row in rows:
            backend_rows.setdefault(row["question_id"], {})[method] = row
            question_info[row["question_id"]] = row
    for qid, methods in backend_rows.items():
        if set(methods) != set(METHODS):
            raise ValueError(f"Question {qid} is not aligned across all official backends")

    labels = []
    for qid in sorted(backend_rows):
        source = question_info[qid]
        label, all_failed = choose_official_silver(backend_rows[qid])
        labels.append(
            {
                "question_id": qid,
                "question": source["question"],
                "question_type": source["question_type"],
                "split": source["split"],
                "silver_label": label,
                "all_failed": all_failed,
            }
        )
    by_split = {
        split: [row for row in labels if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    candidates, best = _tune_router(by_split["train"], by_split["validation"], seed)
    train_validation = by_split["train"] + by_split["validation"]
    fitted_rows = [
        row
        for row in train_validation
        if best["include_all_failed"] or not row["all_failed"]
    ]
    router = build_router(
        best["feature_kind"], best["class_weight"], best["c_value"], seed
    )
    router.fit(
        [row["question"] for row in fitted_rows],
        [row["silver_label"] for row in fitted_rows],
    )
    test = by_split["test"]
    truth = [row["silver_label"] for row in test]
    predictions = router.predict([row["question"] for row in test]).tolist()
    ids = [row["question_id"] for row in test]
    matrix = confusion_matrix(truth, predictions, labels=list(METHODS))
    majority = Counter(row["silver_label"] for row in fitted_rows).most_common(1)[0][0]
    policies = {
        method: _policy(ids, [method] * len(ids), backend_rows) for method in METHODS
    }
    policies["adaptive"] = _policy(ids, predictions, backend_rows)
    policies["oracle"] = _policy(ids, truth, backend_rows)
    summary = {
        "status": "Official backend results; proxy results are excluded.",
        "questions": len(labels),
        "split_counts": dict(Counter(row["split"] for row in labels)),
        "silver_label_distribution": dict(Counter(row["silver_label"] for row in labels)),
        "all_failed": int(sum(row["all_failed"] for row in labels)),
        "router": {
            "family": "TF-IDF + Logistic Regression",
            "selected_on": "validation Macro-F1",
            "selected_hyperparameters": best,
            "test_accuracy": float(accuracy_score(truth, predictions)),
            "test_macro_f1": float(
                f1_score(
                    truth,
                    predictions,
                    labels=list(METHODS),
                    average="macro",
                    zero_division=0,
                )
            ),
            "labels": list(METHODS),
            "confusion_matrix": matrix.tolist(),
            "majority_route": majority,
            "majority_accuracy": float(accuracy_score(truth, [majority] * len(truth))),
            "majority_macro_f1": float(
                f1_score(
                    truth,
                    [majority] * len(truth),
                    labels=list(METHODS),
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "test_policies": policies,
        "comparisons": {
            "adaptive_cost_reduction_vs_pathrag": 1
            - (
                policies["adaptive"]["input_tokens"] + policies["adaptive"]["output_tokens"]
            )
            / max(
                policies["pathrag"]["input_tokens"] + policies["pathrag"]["output_tokens"],
                1e-12,
            ),
            "adaptive_quality_gain_vs_vector": policies["adaptive"]["answer_correctness"]
            - policies["vector"]["answer_correctness"],
        },
        "judge_limitation": (
            "GraphRAG-Bench official metric code was used, but the local judge is the same "
            "Qwen2.5-VL-7B family as the generator; scores may contain self-evaluation bias."
        ),
    }
    with (output_dir / "silver_labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "router_tuning.json").write_text(
        json.dumps(candidates, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    predictions_path = output_dir / "router_test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row, predicted in zip(test, predictions):
            handle.write(
                json.dumps({**row, "predicted_label": predicted}, ensure_ascii=False) + "\n"
            )
    joblib.dump(router, output_dir / "router.joblib")
    _save_plots(summary, matrix, output_dir)
    return summary


def _save_plots(summary: dict, matrix: np.ndarray, output_dir: Path) -> None:
    names = [*METHODS, "adaptive", "oracle"]
    costs = [
        summary["test_policies"][name]["input_tokens"]
        + summary["test_policies"][name]["output_tokens"]
        for name in names
    ]
    quality = [summary["test_policies"][name]["answer_correctness"] for name in names]
    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(costs, quality)
    for name, x_value, y_value in zip(names, costs, quality):
        plt.annotate(name, (x_value, y_value), xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Mean online input + output tokens")
    plt.ylabel("Official answer correctness")
    plt.tight_layout()
    plt.savefig(output_dir / "quality_cost.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.4, 4.7))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(METHODS)), METHODS, rotation=25, ha="right")
    plt.yticks(range(len(METHODS)), METHODS)
    plt.xlabel("Predicted route")
    plt.ylabel("Oracle silver route")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "router_confusion_matrix.png", dpi=180)
    plt.close()

