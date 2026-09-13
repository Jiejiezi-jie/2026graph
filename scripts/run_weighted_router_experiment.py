from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from src.official_analysis import choose_official_silver
from src.official_backends.base import METHODS
from src.official_evaluation import valid_evaluation


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_weight(
    method_rows: dict[str, dict[str, Any]],
    threshold: float,
    margin: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    best_correctness = max(row["answer_correctness"] for row in method_rows.values())
    eligible = [
        method
        for method in METHODS
        if method_rows[method]["answer_correctness"] >= threshold
        and method_rows[method]["answer_correctness"] >= best_correctness - margin
    ]
    if not eligible:
        category = "all_failed"
    elif len(eligible) == 1:
        category = "stable"
    else:
        category = "ambiguous"
    return {
        "category": category,
        "weight": float(weights[category]),
        "eligible_methods": eligible,
        "eligible_count": len(eligible),
        "best_correctness": float(best_correctness),
        "method_correctness": {
            method: float(method_rows[method]["answer_correctness"])
            for method in METHODS
        },
    }


def metrics(
    truth: list[str],
    predicted: list[str],
    sample_weight: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "n": len(truth),
        "accuracy": float(
            accuracy_score(truth, predicted, sample_weight=sample_weight)
        ),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(METHODS),
                average="macro",
                zero_division=0,
                sample_weight=sample_weight,
            )
        ),
        "confusion_matrix": confusion_matrix(
            truth,
            predicted,
            labels=list(METHODS),
            sample_weight=sample_weight,
        ).tolist(),
        "prediction_distribution": dict(Counter(predicted)),
    }


def estimator(config: dict[str, Any], c_value: float, class_weight: str | None) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=int(config["logistic_regression"]["max_iter"]),
        random_state=int(config["seed"]),
    )


def fit_estimator(
    model: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray | None,
) -> LogisticRegression:
    if sample_weights is None:
        return model.fit(features, labels)
    return model.fit(features, labels, sample_weight=sample_weights)


def oof_for_candidate(
    config: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_value: float,
    class_weight: str | None,
    weighted: bool,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    predictions = np.empty(len(labels), dtype=object)
    probabilities = np.zeros((len(labels), len(METHODS)), dtype=float)
    fold_scores = []
    for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
        model = estimator(config, c_value, class_weight)
        fit_estimator(
            model,
            features[train_index],
            labels[train_index],
            sample_weights[train_index] if weighted else None,
        )
        fold_predictions = model.predict(features[valid_index]).tolist()
        predictions[valid_index] = fold_predictions
        raw_probabilities = model.predict_proba(features[valid_index])
        class_to_column = {label: index for index, label in enumerate(model.classes_)}
        probabilities[valid_index] = np.stack(
            [raw_probabilities[:, class_to_column[method]] for method in METHODS],
            axis=1,
        )
        fold_scores.append(
            {
                "fold": fold_number,
                "unweighted": metrics(
                    labels[valid_index].tolist(), fold_predictions
                ),
                "sample_weighted": metrics(
                    labels[valid_index].tolist(),
                    fold_predictions,
                    sample_weights[valid_index],
                ),
            }
        )
    prediction_list = predictions.tolist()
    report = {
        "C": c_value,
        "class_weight": class_weight,
        "fold_scores": fold_scores,
        "cross_validation": metrics(labels.tolist(), prediction_list),
        "sample_weighted_cross_validation": metrics(
            labels.tolist(), prediction_list, sample_weights
        ),
    }
    return report, prediction_list, probabilities


def tune(
    config: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    weighted: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], np.ndarray]:
    candidates = []
    candidate_outputs = []
    for class_weight in config["logistic_regression"]["class_weight_candidates"]:
        for c_value in config["logistic_regression"]["C_candidates"]:
            report, predictions, probabilities = oof_for_candidate(
                config,
                features,
                labels,
                sample_weights,
                folds,
                float(c_value),
                class_weight,
                weighted,
            )
            candidates.append(report)
            candidate_outputs.append((predictions, probabilities))
    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["cross_validation"]["macro_f1"],
            candidates[index]["cross_validation"]["accuracy"],
        ),
    )
    return (
        candidates,
        candidates[best_index],
        candidate_outputs[best_index][0],
        candidate_outputs[best_index][1],
    )


def route_policy(
    routes: list[tuple[str, str]],
    method_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    selected = [method_rows[qid][method] for qid, method in routes]
    return {
        "n": len(selected),
        "answer_correctness": float(
            np.mean([row["answer_correctness"] for row in selected])
        ),
        "evidence_recall": float(
            np.mean([row["evidence_recall"] for row in selected])
        ),
        "input_tokens": float(np.mean([row["input_tokens"] for row in selected])),
        "output_tokens": float(np.mean([row["output_tokens"] for row in selected])),
        "latency_ms": float(np.mean([row["total_time_ms"] for row in selected])),
        "route_distribution": dict(Counter(method for _, method in routes)),
    }


def write_predictions(
    path: Path,
    ids: list[str],
    questions: list[str],
    truth: list[str],
    predictions: list[str],
    probabilities: np.ndarray,
    diagnostics: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, qid in enumerate(ids):
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": questions[index],
                        "silver_label": truth[index],
                        "predicted_label": predictions[index],
                        "probabilities": {
                            method: float(probabilities[index, column])
                            for column, method in enumerate(METHODS)
                        },
                        "weight_category": diagnostics[qid]["category"],
                        "sample_weight": diagnostics[qid]["weight"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed sample weighting against the unweighted BGE-M3 router")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/router_sample_weighting.json")
    )
    args = parser.parse_args()
    started = time.monotonic()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = {name: resolve(path) for name, path in config["inputs"].items()}
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = {
        method: load_jsonl(inputs[method])
        for method in METHODS
    }
    if any(len(method_rows) != 120 for method_rows in rows.values()):
        raise ValueError("Every method must contain exactly 120 evaluated rows")
    if any(not valid_evaluation(row) for method_rows in rows.values() for row in method_rows):
        raise ValueError("All method rows must contain valid official evaluations")
    method_rows = {
        qid: {
            method: next(row for row in rows[method] if row["question_id"] == qid)
            for method in METHODS
        }
        for qid in sorted(row["question_id"] for row in rows["vector"])
    }
    if any(set(row["question_id"] for row in rows[method]) != set(method_rows) for method in METHODS):
        raise ValueError("Method question IDs are not aligned")

    labels = load_jsonl(inputs["labels"])
    labels_by_id = {row["question_id"]: row for row in labels}
    if len(labels_by_id) != 120 or set(labels_by_id) != set(method_rows):
        raise ValueError("Expected 120 aligned unique silver labels")
    threshold = float(config["correctness_threshold"])
    margin = float(config["best_margin"])
    diagnostics = {}
    for qid, per_method in method_rows.items():
        expected_label, expected_all_failed = choose_official_silver(
            per_method, correctness_threshold=threshold, best_margin=margin
        )
        source = labels_by_id[qid]
        if (source["silver_label"], source["all_failed"]) != (
            expected_label,
            expected_all_failed,
        ):
            raise ValueError(f"{qid}: cached label does not match configured rule")
        diagnostics[qid] = classify_weight(
            per_method, threshold, margin, config["weights"]
        )

    embedding_data = np.load(inputs["embeddings"])
    embedding_map = {
        str(qid): vector
        for qid, vector in zip(
            embedding_data["question_ids"].tolist(), embedding_data["embeddings"]
        )
    }
    if not set(method_rows).issubset(embedding_map):
        raise ValueError("Embedding cache is missing questions")

    development_ids = [
        qid for qid in method_rows if labels_by_id[qid]["split"] in ("train", "validation")
    ]
    test_ids = [qid for qid in method_rows if labels_by_id[qid]["split"] == "test"]
    if (len(development_ids), len(test_ids)) != (96, 24):
        raise ValueError("Expected 96 development and 24 test rows")
    development_questions = [labels_by_id[qid]["question"] for qid in development_ids]
    test_questions = [labels_by_id[qid]["question"] for qid in test_ids]
    development_x = np.stack([embedding_map[qid] for qid in development_ids])
    test_x = np.stack([embedding_map[qid] for qid in test_ids])
    development_y = np.array([labels_by_id[qid]["silver_label"] for qid in development_ids])
    test_y = [labels_by_id[qid]["silver_label"] for qid in test_ids]
    development_weights = np.array([diagnostics[qid]["weight"] for qid in development_ids])
    strata = np.array([labels_by_id[qid]["question_type"] for qid in development_ids])
    folds = list(
        StratifiedKFold(
            n_splits=int(config["folds"]), shuffle=True, random_state=int(config["seed"])
        ).split(development_x, strata)
    )

    experiments = {}
    saved = {}
    for name, weighted in (("unweighted", False), ("weighted", True)):
        candidates, best, oof_predictions, oof_probabilities = tune(
            config,
            development_x,
            development_y,
            development_weights,
            folds,
            weighted,
        )
        final_model = estimator(config, best["C"], best["class_weight"])
        fit_estimator(
            final_model,
            development_x,
            development_y,
            development_weights if weighted else None,
        )
        test_predictions = final_model.predict(test_x).tolist()
        raw_test_probabilities = final_model.predict_proba(test_x)
        class_to_column = {label: index for index, label in enumerate(final_model.classes_)}
        test_probabilities = np.stack(
            [raw_test_probabilities[:, class_to_column[method]] for method in METHODS],
            axis=1,
        )
        adaptive = route_policy(
            list(zip(test_ids, test_predictions)), method_rows
        )
        oracle = route_policy(
            [(qid, labels_by_id[qid]["silver_label"]) for qid in test_ids], method_rows
        )
        experiments[name] = {
            "sample_weighting_enabled": weighted,
            "selected_hyperparameters": {
                "C": best["C"], "class_weight": best["class_weight"]
            },
            "cross_validation": best["cross_validation"],
            "sample_weighted_cross_validation": best[
                "sample_weighted_cross_validation"
            ],
            "test": metrics(test_y, test_predictions),
            "adaptive_policy": adaptive,
            "oracle_policy": oracle,
        }
        (output_dir / f"{name}_tuning.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_predictions(
            output_dir / f"{name}_oof_predictions.jsonl",
            development_ids,
            development_questions,
            development_y.tolist(),
            oof_predictions,
            oof_probabilities,
            diagnostics,
        )
        write_predictions(
            output_dir / f"{name}_test_predictions.jsonl",
            test_ids,
            test_questions,
            test_y,
            test_predictions,
            test_probabilities,
            diagnostics,
        )
        joblib.dump(final_model, output_dir / f"{name}_router.joblib")
        saved[name] = best

    baseline = json.loads(inputs["baseline_summary"].read_text(encoding="utf-8"))
    recomputed = experiments["unweighted"]
    for section, metric_name in (("cross_validation", "cross_validation"), ("test", "test")):
        for metric in ("accuracy", "macro_f1"):
            if not np.isclose(
                recomputed[section][metric], baseline[metric_name][metric], atol=1e-12
            ):
                raise ValueError(f"Recomputed unweighted {section} {metric} differs from baseline")

    category_counts = {
        split: dict(
            Counter(
                diagnostics[qid]["category"]
                for qid in method_rows
                if split == "all" or labels_by_id[qid]["split"] == split
            )
        )
        for split in ("train", "validation", "test", "all")
    }
    summary = {
        "protocol": {
            "change_under_test": "fixed sample weighting only",
            "unchanged": [
                "120 questions and 96/24 development/test split",
                "silver labels (threshold=0.60, best_margin=0.10)",
                "BGE-M3 embeddings",
                "question-type-stratified five folds",
                "LogisticRegression candidate grid and seed",
            ],
            "weight_definition": config["weights"],
            "category_definition": {
                "stable": "exactly one eligible method",
                "ambiguous": "two or three eligible methods",
                "all_failed": "no method reaches the threshold and margin rule",
            },
            "category_counts": category_counts,
            "input_sha256": {name: sha256(path) for name, path in inputs.items()},
            "config_sha256": sha256(config_path),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "experiments": experiments,
        "delta_weighted_minus_unweighted": {
            "cross_validation_accuracy": experiments["weighted"]["cross_validation"]["accuracy"]
            - experiments["unweighted"]["cross_validation"]["accuracy"],
            "cross_validation_macro_f1": experiments["weighted"]["cross_validation"]["macro_f1"]
            - experiments["unweighted"]["cross_validation"]["macro_f1"],
            "test_accuracy": experiments["weighted"]["test"]["accuracy"]
            - experiments["unweighted"]["test"]["accuracy"],
            "test_macro_f1": experiments["weighted"]["test"]["macro_f1"]
            - experiments["unweighted"]["test"]["macro_f1"],
            "adaptive_answer_correctness": experiments["weighted"]["adaptive_policy"]["answer_correctness"]
            - experiments["unweighted"]["adaptive_policy"]["answer_correctness"],
            "adaptive_evidence_recall": experiments["weighted"]["adaptive_policy"]["evidence_recall"]
            - experiments["unweighted"]["adaptive_policy"]["evidence_recall"],
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "sample_weights.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "question_id": qid,
                    "split": labels_by_id[qid]["split"],
                    "silver_label": labels_by_id[qid]["silver_label"],
                    **diagnostics[qid],
                },
                ensure_ascii=False,
            )
            + "\n"
            for qid in method_rows
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
