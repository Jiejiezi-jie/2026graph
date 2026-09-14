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

from src.router.official_analysis import choose_official_silver
from src.backend.common.base import METHODS
from src.backend.common.official_evaluation import valid_evaluation
from src.router import build_router


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 2 x 3 label-rule grid with TF-IDF and BGE-M3."
    )
    parser.add_argument("--vector", type=Path, default=Path("result/api/p1/vector_evaluated.jsonl"))
    parser.add_argument("--lightrag", type=Path, default=Path("result/api/p1/lightrag_evaluated.jsonl"))
    parser.add_argument(
        "--pathrag",
        type=Path,
        default=Path("result/shared_current_main/p1/pathrag_evaluated.jsonl"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(
            "result/shared_current_main/p1/analysis_inputs/"
            "question_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--lightrag-graph",
        type=Path,
        default=Path("result/api/indexes/lightrag/medical/graph_chunk_entity_relation.graphml"),
    )
    parser.add_argument(
        "--pathrag-graph",
        type=Path,
        default=Path(
            "result/shared_current_main/indexes/pathrag/"
            "graph_chunk_entity_relation.graphml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/shared_current_main/p1/analysis_threshold_margin_grid"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allowed-evaluation-protocol",
        action="append",
        default=[],
        help="Explicitly permit a known protocol hash; repeat for each allowed hash.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "prediction_distribution": dict(Counter(predicted)),
    }


def validate_inputs(
    paths: dict[str, Path],
    rows: dict[str, list[dict[str, Any]]],
    allowed_protocols: set[str] | None = None,
) -> tuple[list[str], str]:
    if any(len(method_rows) != 120 for method_rows in rows.values()):
        raise ValueError(
            "Expected exactly 120 evaluated questions per method; got "
            + repr({method: len(method_rows) for method, method_rows in rows.items()})
        )
    ids_by_method = {
        method: [row["question_id"] for row in method_rows]
        for method, method_rows in rows.items()
    }
    if any(len(set(ids)) != len(ids) for ids in ids_by_method.values()):
        raise ValueError("Duplicate question IDs found")
    reference_ids = ids_by_method["vector"]
    if any(set(ids) != set(reference_ids) for ids in ids_by_method.values()):
        raise ValueError("Question IDs are not aligned across methods")
    invalid = {
        method: [row["question_id"] for row in method_rows if not valid_evaluation(row)]
        for method, method_rows in rows.items()
    }
    if any(invalid.values()):
        raise ValueError(f"Invalid official evaluation rows: {invalid}")
    protocols = {
        row["evaluation"]["protocol"]
        for method_rows in rows.values()
        for row in method_rows
    }
    if len(protocols) != 1 and protocols != (allowed_protocols or set()):
        raise ValueError(f"Evaluation protocols differ: {sorted(protocols)}")

    by_method = {
        method: {row["question_id"]: row for row in method_rows}
        for method, method_rows in rows.items()
    }
    for qid in reference_ids:
        reference = by_method["vector"][qid]
        for method in METHODS[1:]:
            candidate = by_method[method][qid]
            for field in ("question", "question_type", "ground_truth", "evidence", "split"):
                if candidate.get(field) != reference.get(field):
                    raise ValueError(f"{qid}: inconsistent {field} for {method}")
    # Match analyze_official's canonical ordering so the fixed 5-fold split is
    # comparable with earlier audited router experiments.
    protocol = (
        next(iter(protocols))
        if len(protocols) == 1
        else "explicitly-allowed:" + ",".join(sorted(protocols))
    )
    return sorted(reference_ids), protocol


def build_labels(
    ordered_ids: list[str],
    rows_by_id: dict[str, dict[str, dict[str, Any]]],
    threshold: float,
    margin: float,
) -> list[dict[str, Any]]:
    labels = []
    for qid in ordered_ids:
        method_rows = rows_by_id[qid]
        label, all_failed = choose_official_silver(
            method_rows,
            correctness_threshold=threshold,
            best_margin=margin,
        )
        source = method_rows["vector"]
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
    return labels


def evaluate_candidates(
    feature: str,
    questions: np.ndarray,
    vectors: np.ndarray,
    targets: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_kinds = ("word", "word_char") if feature == "tfidf" else ("bge_m3_cls",)
    c_values = (0.5, 1.0, 2.0) if feature == "tfidf" else (0.01, 0.05, 0.1, 0.5, 1.0)
    candidates = []
    for feature_kind in feature_kinds:
        for class_weight in (None, "balanced"):
            for c_value in c_values:
                all_truth: list[str] = []
                all_predictions: list[str] = []
                fold_scores = []
                for train_index, valid_index in folds:
                    if feature == "tfidf":
                        model = build_router(feature_kind, class_weight, c_value, seed)
                        model.fit(questions[train_index].tolist(), targets[train_index].tolist())
                        predicted = model.predict(questions[valid_index].tolist()).tolist()
                    else:
                        model = LogisticRegression(
                            C=c_value,
                            class_weight=class_weight,
                            max_iter=2_000,
                            random_state=seed,
                        )
                        model.fit(vectors[train_index], targets[train_index])
                        predicted = model.predict(vectors[valid_index]).tolist()
                    truth = targets[valid_index].tolist()
                    all_truth.extend(truth)
                    all_predictions.extend(predicted)
                    fold_scores.append(score(truth, predicted))
                candidates.append(
                    {
                        "feature_kind": feature_kind,
                        "class_weight": class_weight,
                        "C": c_value,
                        "fold_scores": fold_scores,
                        "cross_validation": score(all_truth, all_predictions),
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
    started = time.monotonic()
    args = parse_args()
    paths = {method: resolve(getattr(args, method)) for method in METHODS}
    rows = {method: load_jsonl(path) for method, path in paths.items()}
    ordered_ids, evaluation_protocol = validate_inputs(
        paths, rows, set(args.allowed_evaluation_protocol)
    )

    lightrag_graph = resolve(args.lightrag_graph)
    pathrag_graph = resolve(args.pathrag_graph)
    graph_hash = sha256(lightrag_graph)
    if sha256(pathrag_graph) != graph_hash:
        raise ValueError("LightRAG and PathRAG graph files are not byte-identical")

    rows_by_id = {
        qid: {
            method: next(row for row in rows[method] if row["question_id"] == qid)
            for method in METHODS
        }
        for qid in ordered_ids
    }
    embedding_data = np.load(resolve(args.embeddings))
    embedding_map = {
        str(qid): vector
        for qid, vector in zip(
            embedding_data["question_ids"].tolist(),
            embedding_data["embeddings"],
        )
    }
    if not set(ordered_ids).issubset(embedding_map):
        raise ValueError("BGE-M3 embedding cache does not contain all 120 questions")

    development_ids = [
        qid for qid in ordered_ids if rows_by_id[qid]["vector"]["split"] in ("train", "validation")
    ]
    test_ids = [qid for qid in ordered_ids if rows_by_id[qid]["vector"]["split"] == "test"]
    if (len(development_ids), len(test_ids)) != (96, 24):
        raise ValueError(f"Expected 96 development and 24 test questions, got {len(development_ids)}/{len(test_ids)}")
    questions = np.array(
        [rows_by_id[qid]["vector"]["question"] for qid in development_ids],
        dtype=object,
    )
    test_questions = [rows_by_id[qid]["vector"]["question"] for qid in test_ids]
    vectors = np.stack([embedding_map[qid] for qid in development_ids])
    test_vectors = np.stack([embedding_map[qid] for qid in test_ids])
    strata = np.array([rows_by_id[qid]["vector"]["question_type"] for qid in development_ids])
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(
            questions, strata
        )
    )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for threshold in (0.50, 0.60):
        for margin in (0.05, 0.10, 0.15):
            labels = build_labels(ordered_ids, rows_by_id, threshold, margin)
            labels_by_id = {row["question_id"]: row for row in labels}
            targets = np.array([labels_by_id[qid]["silver_label"] for qid in development_ids])
            test_targets = [labels_by_id[qid]["silver_label"] for qid in test_ids]
            for feature in ("tfidf", "bge_m3"):
                candidates, best = evaluate_candidates(
                    feature, questions, vectors, targets, folds, args.seed
                )
                if feature == "tfidf":
                    model = build_router(
                        best["feature_kind"], best["class_weight"], best["C"], args.seed
                    )
                    model.fit(questions.tolist(), targets.tolist())
                    predicted = model.predict(test_questions).tolist()
                else:
                    model = LogisticRegression(
                        C=best["C"],
                        class_weight=best["class_weight"],
                        max_iter=2_000,
                        random_state=args.seed,
                    )
                    model.fit(vectors, targets)
                    predicted = model.predict(test_vectors).tolist()

                name = f"threshold_{threshold:.2f}_margin_{margin:.2f}_{feature}"
                target_dir = output_dir / name
                target_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "correctness_threshold": threshold,
                    "best_margin": margin,
                    "feature": feature,
                    "development_count": len(development_ids),
                    "test_count": len(test_ids),
                    "development_label_distribution": dict(Counter(targets.tolist())),
                    "test_label_distribution": dict(Counter(test_targets)),
                    "all_failed": int(sum(row["all_failed"] for row in labels)),
                    "cross_validation": best["cross_validation"],
                    "test": score(test_targets, predicted),
                    "majority_accuracy": max(Counter(test_targets).values()) / len(test_targets),
                    "selected_hyperparameters": {
                        "feature_kind": best["feature_kind"],
                        "class_weight": best["class_weight"],
                        "C": best["C"],
                    },
                }
                results.append(report)
                (target_dir / "summary.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (target_dir / "tuning.json").write_text(
                    json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                with (target_dir / "silver_labels.jsonl").open("w", encoding="utf-8") as handle:
                    for row in labels:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                with (target_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
                    for qid, truth, prediction in zip(test_ids, test_targets, predicted):
                        handle.write(
                            json.dumps(
                                {
                                    "question_id": qid,
                                    "question": rows_by_id[qid]["vector"]["question"],
                                    "silver_label": truth,
                                    "predicted_label": prediction,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                joblib.dump(model, target_dir / "router.joblib")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    summary = {
        "protocol": {
            "questions": 120,
            "development_count": 96,
            "test_count": 24,
            "folds": 5,
            "fold_stratification": "question_type",
            "seed": args.seed,
            "thresholds": [0.50, 0.60],
            "best_margins": [0.05, 0.10, 0.15],
            "features": ["tfidf", "bge_m3"],
            "all_failed_policy": "included as fallback method label",
            "hyperparameter_selection": "OOF Macro-F1, then OOF accuracy",
            "evaluation_protocol": evaluation_protocol,
            "git_commit": commit,
            "shared_graph_sha256": graph_hash,
            "shared_graph_verified_byte_identical": True,
            "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()},
            "embedding_cache": str(resolve(args.embeddings).relative_to(ROOT)),
        },
        "results": results,
        "analysis_elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
