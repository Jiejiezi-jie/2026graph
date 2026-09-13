from __future__ import annotations

import argparse
import asyncio
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_shared_graph_rule_grid import evaluate_candidates, score
from src.official_analysis import choose_official_silver
from src.official_backends.base import METHODS
from src.official_backends.model_client import build_embedding_client
from src.official_evaluation import valid_evaluation
from src.router import build_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the fixed 96-row and expanded 276-row router training sets."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/router_data_expansion.json")
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
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


def route_policy(
    ids: list[str], routes: list[str], rows_by_id: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    selected = [rows_by_id[qid][route] for qid, route in zip(ids, routes)]
    return {
        "n": len(selected),
        "answer_correctness": float(np.mean([row["answer_correctness"] for row in selected])),
        "evidence_recall": float(np.mean([row["evidence_recall"] for row in selected])),
        "input_tokens": float(np.mean([row["input_tokens"] for row in selected])),
        "output_tokens": float(np.mean([row["output_tokens"] for row in selected])),
        "latency_ms": float(np.mean([row["total_time_ms"] for row in selected])),
        "route_distribution": dict(Counter(routes)),
    }


def validate_and_merge(
    config: dict[str, Any]
) -> tuple[dict[str, dict[str, dict[str, Any]]], set[str], set[str], str, dict[str, str]]:
    paths: dict[str, dict[str, Path]] = {
        "baseline": {
            method: resolve(config["baseline_inputs"][method]) for method in METHODS
        },
        "expansion": {
            method: resolve(config["results_dir"]) / "p1" / f"{method}_evaluated.jsonl"
            for method in METHODS
        },
    }
    collections = {
        group: {method: load_jsonl(path) for method, path in method_paths.items()}
        for group, method_paths in paths.items()
    }
    expected = {"baseline": 120, "expansion": 180}
    for group, methods in collections.items():
        counts = {method: len(rows) for method, rows in methods.items()}
        if any(count != expected[group] for count in counts.values()):
            raise ValueError(f"{group}: unexpected method counts {counts}")
        if any(not valid_evaluation(row) for rows in methods.values() for row in rows):
            raise ValueError(f"{group}: all rows must have valid official evaluations")

    merged: dict[str, dict[str, dict[str, Any]]] = {}
    group_ids: dict[str, set[str]] = {}
    protocols: set[str] = set()
    input_hashes: dict[str, str] = {}
    for group, methods in collections.items():
        ids_by_method = {
            method: {row["question_id"] for row in rows}
            for method, rows in methods.items()
        }
        if len({frozenset(ids) for ids in ids_by_method.values()}) != 1:
            raise ValueError(f"{group}: question IDs differ across methods")
        group_ids[group] = ids_by_method["vector"]
        for method, rows in methods.items():
            input_hashes[f"{group}_{method}"] = sha256(paths[group][method])
            for row in rows:
                protocols.add(row["evaluation"]["protocol"])
                merged.setdefault(row["question_id"], {})[method] = row

    if group_ids["baseline"].intersection(group_ids["expansion"]):
        raise ValueError("Expansion overlaps baseline IDs")
    allowed_protocols = set(config.get("allowed_evaluation_protocols", []))
    if len(protocols) != 1 and protocols != allowed_protocols:
        raise ValueError(f"Evaluation protocol mismatch: {sorted(protocols)}")
    if any(len(method_rows) != len(METHODS) for method_rows in merged.values()):
        raise ValueError("At least one question is missing a method")

    for qid, method_rows in merged.items():
        reference = method_rows["vector"]
        for method in METHODS[1:]:
            candidate = method_rows[method]
            for field in ("question", "question_type", "ground_truth", "evidence", "split"):
                if candidate.get(field) != reference.get(field):
                    raise ValueError(f"{qid}: {method} differs on {field}")
    if any(merged[qid]["vector"]["split"] == "test" for qid in group_ids["expansion"]):
        raise ValueError("Expansion must not contain test rows")
    baseline_test = {
        qid for qid in group_ids["baseline"] if merged[qid]["vector"]["split"] == "test"
    }
    if len(baseline_test) != 24:
        raise ValueError(f"Expected frozen 24-row test, got {len(baseline_test)}")
    protocol = (
        next(iter(protocols))
        if len(protocols) == 1
        else "explicitly-allowed:" + ",".join(sorted(protocols))
    )
    return merged, group_ids["baseline"], group_ids["expansion"], protocol, input_hashes


def build_labels(
    rows_by_id: dict[str, dict[str, dict[str, Any]]], threshold: float, margin: float
) -> dict[str, dict[str, Any]]:
    labels = {}
    for qid, method_rows in rows_by_id.items():
        label, all_failed = choose_official_silver(
            method_rows, correctness_threshold=threshold, best_margin=margin
        )
        source = method_rows["vector"]
        labels[qid] = {
            "question_id": qid,
            "question": source["question"],
            "question_type": source["question_type"],
            "split": source["split"],
            "silver_label": label,
            "all_failed": all_failed,
        }
    return labels


async def add_missing_embeddings(
    config: dict[str, Any], ids: list[str], labels: dict[str, dict[str, Any]],
    existing: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    missing = [qid for qid in ids if qid not in existing]
    if not missing:
        return existing
    embedding = build_embedding_client(config["embedding"])
    try:
        vectors = await embedding.embed([labels[qid]["question"] for qid in missing])
    finally:
        embedding.unload()
    return existing | {qid: vector for qid, vector in zip(missing, vectors)}


def load_or_build_embeddings(
    config: dict[str, Any], ids: list[str], labels: dict[str, dict[str, Any]], output_dir: Path
) -> tuple[dict[str, np.ndarray], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "question_embeddings.npz"
    existing: dict[str, np.ndarray] = {}
    if cache.exists():
        data = np.load(cache)
        existing = {
            str(qid): vector
            for qid, vector in zip(data["question_ids"].tolist(), data["embeddings"])
        }
    else:
        baseline_cache = (
            ROOT
            / "results_shared_lightrag_current_main/p1/analysis_inputs/question_embeddings.npz"
        )
        if baseline_cache.exists():
            data = np.load(baseline_cache)
            existing = {
                str(qid): vector
                for qid, vector in zip(data["question_ids"].tolist(), data["embeddings"])
            }
    embedding_map = asyncio.run(add_missing_embeddings(config, ids, labels, existing))
    if not set(ids).issubset(embedding_map):
        raise AssertionError("Embedding cache is incomplete")
    ordered = sorted(ids)
    np.savez_compressed(
        cache,
        question_ids=np.asarray(ordered),
        embeddings=np.stack([embedding_map[qid] for qid in ordered]),
    )
    return embedding_map, cache


def run_one(
    feature: str,
    development_ids: list[str],
    test_ids: list[str],
    labels: dict[str, dict[str, Any]],
    embedding_map: dict[str, np.ndarray],
    rows_by_id: dict[str, dict[str, dict[str, Any]]],
    seed: int,
    target_dir: Path,
) -> dict[str, Any]:
    questions = np.asarray([labels[qid]["question"] for qid in development_ids], dtype=object)
    targets = np.asarray([labels[qid]["silver_label"] for qid in development_ids])
    vectors = np.stack([embedding_map[qid] for qid in development_ids])
    strata = np.asarray([labels[qid]["question_type"] for qid in development_ids])
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(questions, strata)
    )
    candidates, best = evaluate_candidates(
        feature, questions, vectors, targets, folds, seed
    )
    test_questions = [labels[qid]["question"] for qid in test_ids]
    if feature == "tfidf":
        model = build_router(best["feature_kind"], best["class_weight"], best["C"], seed)
        model.fit(questions.tolist(), targets.tolist())
        predictions = model.predict(test_questions).tolist()
    else:
        model = LogisticRegression(
            C=best["C"], class_weight=best["class_weight"], max_iter=2_000,
            random_state=seed,
        )
        model.fit(vectors, targets)
        test_vectors = np.stack([embedding_map[qid] for qid in test_ids])
        predictions = model.predict(test_vectors).tolist()
    truth = [labels[qid]["silver_label"] for qid in test_ids]
    report = {
        "feature": feature,
        "development_count": len(development_ids),
        "test_count": len(test_ids),
        "development_label_distribution": dict(Counter(targets.tolist())),
        "test_label_distribution": dict(Counter(truth)),
        "development_all_failed": sum(labels[qid]["all_failed"] for qid in development_ids),
        "cross_validation": best["cross_validation"],
        "test": score(truth, predictions),
        "selected_hyperparameters": {
            "feature_kind": best["feature_kind"],
            "class_weight": best["class_weight"],
            "C": best["C"],
        },
        "test_policy": route_policy(test_ids, predictions, rows_by_id),
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (target_dir / "tuning.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (target_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for qid, target, predicted in zip(test_ids, truth, predictions):
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": labels[qid]["question"],
                        "silver_label": target,
                        "predicted_label": predicted,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    joblib.dump(model, target_dir / "router.joblib")
    return report


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows_by_id, baseline_ids, expansion_ids, protocol, input_hashes = validate_and_merge(config)
    labels = build_labels(
        rows_by_id,
        float(config["correctness_threshold"]),
        float(config["best_margin"]),
    )
    test_ids = sorted(
        qid for qid in baseline_ids if labels[qid]["split"] == "test"
    )
    baseline_development = sorted(baseline_ids - set(test_ids))
    expanded_development = sorted(set(baseline_development) | expansion_ids)
    if (len(baseline_development), len(expanded_development), len(test_ids)) != (96, 276, 24):
        raise AssertionError("Unexpected before/after development or frozen test counts")

    output_dir = resolve(config["results_dir"]) / "p1" / "router_data_expansion"
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_map, embedding_cache = load_or_build_embeddings(
        config, sorted(rows_by_id), labels, output_dir
    )
    experiments: dict[str, dict[str, Any]] = {}
    for feature in ("tfidf", "bge_m3"):
        experiments[f"baseline_{feature}"] = run_one(
            feature, baseline_development, test_ids, labels, embedding_map, rows_by_id,
            int(config["seed"]), output_dir / f"baseline_{feature}",
        )
        experiments[f"expanded_{feature}"] = run_one(
            feature, expanded_development, test_ids, labels, embedding_map, rows_by_id,
            int(config["seed"]), output_dir / f"expanded_{feature}",
        )

    deltas = {}
    for feature in ("tfidf", "bge_m3"):
        before = experiments[f"baseline_{feature}"]
        after = experiments[f"expanded_{feature}"]
        deltas[feature] = {
            "oof_accuracy": after["cross_validation"]["accuracy"] - before["cross_validation"]["accuracy"],
            "oof_macro_f1": after["cross_validation"]["macro_f1"] - before["cross_validation"]["macro_f1"],
            "test_accuracy": after["test"]["accuracy"] - before["test"]["accuracy"],
            "test_macro_f1": after["test"]["macro_f1"] - before["test"]["macro_f1"],
            "test_answer_correctness": after["test_policy"]["answer_correctness"] - before["test_policy"]["answer_correctness"],
            "test_evidence_recall": after["test_policy"]["evidence_recall"] - before["test_policy"]["evidence_recall"],
        }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    summary = {
        "status": "fixed data-size ablation; frozen test set was not used for selection",
        "configuration": {
            "correctness_threshold": config["correctness_threshold"],
            "best_margin": config["best_margin"],
            "seed": config["seed"],
            "folds": 5,
            "fold_stratification": "question_type",
            "baseline_development_count": 96,
            "expanded_development_count": 276,
            "frozen_test_count": 24,
            "added_question_count": 180,
            "added_question_type_distribution": {
                "Fact Retrieval": 90,
                "Complex Reasoning": 90,
            },
            "evaluation_protocol": protocol,
            "embedding_cache": str(embedding_cache.relative_to(ROOT)),
            "git_commit": commit,
        },
        "experiments": experiments,
        "deltas": deltas,
        "input_sha256": input_hashes,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "silver_labels.jsonl").open("w", encoding="utf-8") as handle:
        for qid in sorted(labels):
            handle.write(json.dumps(labels[qid], ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
