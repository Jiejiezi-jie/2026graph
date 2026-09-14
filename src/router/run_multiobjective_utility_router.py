from __future__ import annotations

import argparse
import hashlib
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
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router.analyze_router_data_expansion import build_labels, resolve, validate_and_merge
from src.backend.common.base import METHODS


TARGET_METRICS = ("answer_correctness", "evidence_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a question-only multi-objective utility router."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/router_multiobjective_utility.json"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_matrix(
    ids: list[str], rows_by_id: dict[str, dict[str, dict[str, Any]]]
) -> np.ndarray:
    return np.asarray(
        [
            [
                float(rows_by_id[qid][method][metric])
                for method in METHODS
                for metric in TARGET_METRICS
            ]
            for qid in ids
        ],
        dtype=float,
    )


def route_from_predictions(predictions: np.ndarray, evidence_weight: float) -> np.ndarray:
    clipped = np.clip(predictions, 0.0, 1.0)
    correctness = clipped[:, 0::2]
    evidence = clipped[:, 1::2]
    return np.argmax(correctness + evidence_weight * evidence, axis=1)


def policy_metrics(targets: np.ndarray, routes: np.ndarray) -> dict[str, Any]:
    indexes = np.arange(len(routes))
    correctness = targets[:, 0::2][indexes, routes]
    evidence = targets[:, 1::2][indexes, routes]
    return {
        "n": len(routes),
        "answer_correctness": float(np.mean(correctness)),
        "evidence_recall": float(np.mean(evidence)),
        "joint_harmonic_mean": float(
            2 * np.mean(correctness) * np.mean(evidence)
            / (np.mean(correctness) + np.mean(evidence))
        ),
        "route_distribution": {
            method: int(np.sum(routes == index))
            for index, method in enumerate(METHODS)
        },
    }


def full_policy_metrics(
    ids: list[str],
    routes: np.ndarray,
    rows_by_id: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    selected = [rows_by_id[qid][METHODS[int(route)]] for qid, route in zip(ids, routes)]
    base = policy_metrics(target_matrix(ids, rows_by_id), routes)
    base.update(
        {
            "input_tokens": float(np.mean([row["input_tokens"] for row in selected])),
            "output_tokens": float(np.mean([row["output_tokens"] for row in selected])),
            "latency_ms": float(np.mean([row["total_time_ms"] for row in selected])),
        }
    )
    return base


def load_routes(path: Path, ids: list[str]) -> np.ndarray:
    rows = load_jsonl(path)
    by_id = {row["question_id"]: row["predicted_label"] for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError(f"Route file is not aligned with expected IDs: {path}")
    return np.asarray([METHODS.index(by_id[qid]) for qid in ids], dtype=int)


def classification_metrics(truth: list[str], routes: np.ndarray) -> dict[str, Any]:
    predicted = [METHODS[int(route)] for route in routes]
    return {
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


def paired_bootstrap(
    ids: list[str],
    candidate_routes: np.ndarray,
    reference_routes: np.ndarray,
    rows_by_id: dict[str, dict[str, dict[str, Any]]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(ids), size=(samples, len(ids)))
    result: dict[str, Any] = {"samples": samples, "seed": seed}
    for metric in TARGET_METRICS:
        differences = np.asarray(
            [
                rows_by_id[qid][METHODS[int(candidate_route)]][metric]
                - rows_by_id[qid][METHODS[int(reference_route)]][metric]
                for qid, candidate_route, reference_route in zip(
                    ids, candidate_routes, reference_routes
                )
            ],
            dtype=float,
        )
        bootstrap_means = differences[indexes].mean(axis=1)
        result[metric] = {
            "mean_delta": float(np.mean(differences)),
            "paired_bootstrap_95ci": [
                float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])
            ],
            "questions_better_equal_worse": {
                "better": int(np.sum(differences > 1e-12)),
                "equal": int(np.sum(np.abs(differences) <= 1e-12)),
                "worse": int(np.sum(differences < -1e-12)),
            },
        }
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
        raise AssertionError("Expected 276 development, 24 frozen test, and 300 total rows")

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
    development_y = target_matrix(development_ids, rows_by_id)
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

    reference_oof_path = resolve(config["reference_300_oof"])
    reference_oof_routes = load_routes(reference_oof_path, development_ids)
    reference_oof = policy_metrics(development_y, reference_oof_routes)

    candidates: list[dict[str, Any]] = []
    oof_by_alpha: dict[float, np.ndarray] = {}
    for alpha_value in config["ridge_alpha_candidates"]:
        alpha = float(alpha_value)
        oof_predictions = np.zeros_like(development_y)
        for train_index, valid_index in folds:
            model = Ridge(alpha=alpha)
            model.fit(development_x[train_index], development_y[train_index])
            oof_predictions[valid_index] = model.predict(development_x[valid_index])
        oof_by_alpha[alpha] = oof_predictions
        for evidence_weight_value in config["evidence_weight_candidates"]:
            evidence_weight = float(evidence_weight_value)
            routes = route_from_predictions(oof_predictions, evidence_weight)
            scores = policy_metrics(development_y, routes)
            gain_correctness = (
                scores["answer_correctness"] - reference_oof["answer_correctness"]
            )
            gain_evidence = scores["evidence_recall"] - reference_oof["evidence_recall"]
            candidates.append(
                {
                    "alpha": alpha,
                    "evidence_weight": evidence_weight,
                    "oof_policy": scores,
                    "delta_vs_reference": {
                        "answer_correctness": gain_correctness,
                        "evidence_recall": gain_evidence,
                    },
                    "feasible": gain_correctness >= 0.0 and gain_evidence >= 0.0,
                    "minimum_gain": min(gain_correctness, gain_evidence),
                    "total_gain": gain_correctness + gain_evidence,
                }
            )

    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    if not feasible:
        raise ValueError("No candidate improves both OOF objectives over the reference")
    selected = max(
        feasible,
        key=lambda candidate: (
            candidate["minimum_gain"],
            candidate["total_gain"],
            -candidate["alpha"],
            -candidate["evidence_weight"],
        ),
    )
    selected_alpha = float(selected["alpha"])
    selected_evidence_weight = float(selected["evidence_weight"])
    selected_oof_predictions = oof_by_alpha[selected_alpha]
    selected_oof_routes = route_from_predictions(
        selected_oof_predictions, selected_evidence_weight
    )

    # Candidate selection is complete above. Test scores are first consumed below.
    final_model = Ridge(alpha=selected_alpha)
    final_model.fit(development_x, development_y)
    test_x = np.stack([embedding_map[qid] for qid in test_ids])
    test_predictions = final_model.predict(test_x)
    test_routes = route_from_predictions(test_predictions, selected_evidence_weight)
    selected_test = full_policy_metrics(test_ids, test_routes, rows_by_id)
    reference_120_routes = load_routes(resolve(config["reference_120_test"]), test_ids)
    reference_300_routes = load_routes(resolve(config["reference_300_test"]), test_ids)
    reference_120_test = full_policy_metrics(test_ids, reference_120_routes, rows_by_id)
    reference_300_test = full_policy_metrics(test_ids, reference_300_routes, rows_by_id)
    fixed_test = {
        method: full_policy_metrics(
            test_ids,
            np.full(len(test_ids), index, dtype=int),
            rows_by_id,
        )
        for index, method in enumerate(METHODS)
    }

    labels = build_labels(rows_by_id, threshold=0.60, margin=0.10)
    oof_truth = [labels[qid]["silver_label"] for qid in development_ids]
    test_truth = [labels[qid]["silver_label"] for qid in test_ids]
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "router.joblib"
    joblib.dump(final_model, model_path)

    with (output_dir / "oof_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, qid in enumerate(development_ids):
            row = np.clip(selected_oof_predictions[index], 0.0, 1.0)
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "split": rows_by_id[qid]["vector"]["split"],
                        "predicted_method_metrics": {
                            method: {
                                metric: float(row[method_index * 2 + metric_index])
                                for metric_index, metric in enumerate(TARGET_METRICS)
                            }
                            for method_index, method in enumerate(METHODS)
                        },
                        "predicted_route": METHODS[int(selected_oof_routes[index])],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, qid in enumerate(test_ids):
            row = np.clip(test_predictions[index], 0.0, 1.0)
            route = METHODS[int(test_routes[index])]
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": rows_by_id[qid]["vector"]["question"],
                        "predicted_method_metrics": {
                            method: {
                                metric: float(row[method_index * 2 + metric_index])
                                for metric_index, metric in enumerate(TARGET_METRICS)
                            }
                            for method_index, method in enumerate(METHODS)
                        },
                        "predicted_route": route,
                        "actual_selected_answer_correctness": float(
                            rows_by_id[qid][route]["answer_correctness"]
                        ),
                        "actual_selected_evidence_recall": float(
                            rows_by_id[qid][route]["evidence_recall"]
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    (output_dir / "candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "completed offline; no retrieval, generation, evaluation, or API calls",
        "protocol": {
            "total_questions": 300,
            "development_count": 276,
            "frozen_test_count": 24,
            "feature": config["feature"],
            "model": "multi-output Ridge regression",
            "model_inputs": ["question BGE-M3 embedding"],
            "development_targets": list(config["target_order"]),
            "inference_utility": "predicted_answer_correctness + evidence_weight * predicted_evidence_recall",
            "folds": config["folds"],
            "fold_stratification": "question_type",
            "seed": config["seed"],
            "candidate_selection": config["selection"],
            "evaluation_protocol": protocol,
            "scikit_learn_version": sklearn.__version__,
            "test_used_for_training_or_selection": False,
            "new_api_calls": 0,
            "input_sha256": input_hashes
            | {
                "embedding_cache": sha256(embedding_path),
                "reference_300_oof": sha256(reference_oof_path),
                "config": sha256(config_path),
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "selected_parameters": {
            "ridge_alpha": selected_alpha,
            "evidence_weight": selected_evidence_weight,
        },
        "development_oof": {
            "reference_300_weighted_hard_router": reference_oof,
            "multiobjective_utility_router": selected["oof_policy"],
            "delta": selected["delta_vs_reference"],
            "classification_against_hard_silver_labels": classification_metrics(
                oof_truth, selected_oof_routes
            ),
        },
        "frozen_test": {
            "fixed_methods": fixed_test,
            "reference_120_weighted_hard_router": reference_120_test,
            "reference_300_weighted_hard_router": reference_300_test,
            "multiobjective_utility_router": selected_test,
            "delta_vs_reference_120": {
                "answer_correctness": selected_test["answer_correctness"]
                - reference_120_test["answer_correctness"],
                "evidence_recall": selected_test["evidence_recall"]
                - reference_120_test["evidence_recall"],
            },
            "paired_bootstrap_vs_reference_120": paired_bootstrap(
                test_ids,
                test_routes,
                reference_120_routes,
                rows_by_id,
                int(config["bootstrap_samples"]),
                int(config["seed"]),
            ),
            "classification_against_hard_silver_labels": classification_metrics(
                test_truth, test_routes
            ),
        },
        "artifacts": {
            "model": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256(model_path),
            "oof_predictions": str(
                (output_dir / "oof_predictions.jsonl").relative_to(ROOT)
            ),
            "test_predictions": str(
                (output_dir / "test_predictions.jsonl").relative_to(ROOT)
            ),
            "candidates": str((output_dir / "candidates.json").relative_to(ROOT)),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
