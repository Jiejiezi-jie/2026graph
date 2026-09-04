from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.backends import OfflineGraphIndex
from src.data import chunk_by_word_window, load_medical_benchmark, select_questions
from src.metrics import choose_silver_label, evidence_scores, rouge_l_f1
from src.router import build_router


METHODS = ("vector", "light_proxy", "path_proxy")
DISPLAY_NAMES = {
    "vector": "Vector proxy",
    "light_proxy": "Light graph proxy",
    "path_proxy": "Path graph proxy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Adaptive GraphRAG phase-one protocol")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def safe_stratify(values: list[str]) -> list[str] | None:
    counts = Counter(values)
    return values if counts and min(counts.values()) >= 2 else None


def split_indices(rows: list[dict], config: dict) -> dict[int, str]:
    indices = np.arange(len(rows))
    strata = [f"{row['question_type']}::{row['silver_label']}" for row in rows]
    train_dev, test = train_test_split(
        indices,
        test_size=config["test_size"],
        random_state=config["seed"],
        stratify=safe_stratify(strata),
    )
    remaining_strata = [strata[index] for index in train_dev]
    train, dev = train_test_split(
        train_dev,
        test_size=config["dev_size_within_train"],
        random_state=config["seed"],
        stratify=safe_stratify(remaining_strata),
    )
    split = {int(index): "train" for index in train}
    split.update({int(index): "dev" for index in dev})
    split.update({int(index): "test" for index in test})
    return split


def metric_block(method_rows: list[dict]) -> dict:
    return {
        "n": len(method_rows),
        "evidence_coverage": float(np.mean([row["evidence_coverage"] for row in method_rows])),
        "evidence_hit_rate": float(np.mean([row["evidence_hit_rate"] for row in method_rows])),
        "answer_proxy_rouge_l": float(np.mean([row["answer_proxy_rouge_l"] for row in method_rows])),
        "context_words": float(np.mean([row["context_words"] for row in method_rows])),
        "latency_ms": float(np.mean([row["latency_ms"] for row in method_rows])),
    }


def evaluate_policy(
    question_rows: list[dict],
    backend_rows: dict[str, dict[str, dict]],
    question_ids: list[str],
    choices: list[str],
) -> dict:
    selected = [backend_rows[question_id][method] for question_id, method in zip(question_ids, choices)]
    result = metric_block(selected)
    result["route_distribution"] = dict(Counter(choices))
    return result


def tune_router(
    train_questions: list[str],
    train_labels: list[str],
    dev_questions: list[str],
    dev_labels: list[str],
    seed: int,
) -> tuple[dict, list[dict]]:
    candidates: list[dict] = []
    for feature_kind in ("word", "word_char"):
        for class_weight in (None, "balanced"):
            for c_value in (0.5, 1.0, 2.0):
                model = build_router(feature_kind, class_weight, c_value, seed)
                model.fit(train_questions, train_labels)
                predictions = model.predict(dev_questions)
                candidates.append(
                    {
                        "feature_kind": feature_kind,
                        "class_weight": class_weight,
                        "c_value": c_value,
                        "macro_f1": float(f1_score(dev_labels, predictions, average="macro", zero_division=0)),
                        "accuracy": float(accuracy_score(dev_labels, predictions)),
                    }
                )
    best = max(candidates, key=lambda row: (row["macro_f1"], row["accuracy"]))
    return best, candidates


def save_plots(summary: dict, confusion: np.ndarray, label_distribution: dict, output_dir: Path) -> None:
    policies = ["vector", "light_proxy", "path_proxy", "adaptive", "oracle"]
    x = [summary["test_policies"][name]["context_words"] for name in policies]
    y = [summary["test_policies"][name]["evidence_coverage"] for name in policies]
    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(x, y, s=70)
    for name, x_value, y_value in zip(policies, x, y):
        plt.annotate(name, (x_value, y_value), xytext=(5, 4), textcoords="offset points")
    plt.xlabel("Average retrieved context words")
    plt.ylabel("Evidence coverage")
    plt.title("Quality-cost trade-off on the held-out test set")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "quality_cost.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    names = list(METHODS)
    counts = [label_distribution.get(name, 0) for name in names]
    plt.bar(names, counts, color=["#64748b", "#3b82f6", "#8b5cf6"])
    plt.ylabel("Questions")
    plt.title("Automatically collected silver labels")
    plt.tight_layout()
    plt.savefig(output_dir / "silver_label_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.4, 4.7))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(METHODS)), METHODS, rotation=25, ha="right")
    plt.yticks(range(len(METHODS)), METHODS)
    plt.xlabel("Predicted route")
    plt.ylabel("Oracle silver route")
    plt.title("Router confusion matrix")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            plt.text(col, row, str(confusion[row, col]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "router_confusion_matrix.png", dpi=180)
    plt.close()


def benchmark_commit(benchmark_dir: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=benchmark_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus, all_questions = load_medical_benchmark(args.benchmark_dir)
    questions = select_questions(
        all_questions,
        set(config["question_types"]),
        config["max_per_type"],
    )
    chunks = chunk_by_word_window(
        corpus,
        config["chunk_words"],
        config["chunk_overlap_words"],
    )
    print(f"Loaded {len(questions)} questions and built {len(chunks)} chunks")
    index = OfflineGraphIndex(
        chunks,
        vector_top_k=config["vector_top_k"],
        light_top_k=config["light_top_k"],
        path_top_k=config["path_top_k"],
        graph_neighbors=config["graph_neighbors"],
        graph_similarity_floor=config["graph_similarity_floor"],
        adjacent_edge_weight=config["adjacent_edge_weight"],
    )

    backend_rows: dict[str, dict[str, dict]] = {}
    flat_backend_rows: list[dict] = []
    question_rows: list[dict] = []
    for position, question in enumerate(questions, start=1):
        per_method: dict[str, dict] = {}
        for method in METHODS:
            retrieval = index.retrieve(question["question"], method)
            metrics = evidence_scores(
                question["evidence"],
                retrieval.contexts,
                config["evidence_hit_threshold"],
            )
            row = {
                "id": question["id"],
                "question_type": question["question_type"],
                "method": method,
                **metrics,
                "answer_proxy_rouge_l": rouge_l_f1(question["answer"], retrieval.answer_proxy),
                "context_words": retrieval.context_words,
                "latency_ms": retrieval.latency_ms,
                "retrieved_chunk_ids": json.dumps(retrieval.chunk_ids),
                "trace": json.dumps(retrieval.trace),
                "answer_proxy": retrieval.answer_proxy,
            }
            per_method[method] = row
            flat_backend_rows.append(row)
        backend_rows[question["id"]] = per_method
        silver_label = choose_silver_label(per_method, config["route_quality_tolerance"])
        best_quality = max(row["evidence_coverage"] for row in per_method.values())
        question_rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "answer": question["answer"],
                "evidence": question["evidence"],
                "evidence_relations": question["evidence_relations"],
                "question_type": question["question_type"],
                "silver_label": silver_label,
                "best_evidence_coverage": best_quality,
                "low_signal": best_quality < 0.25,
            }
        )
        if position % 250 == 0:
            print(f"Processed {position}/{len(questions)} questions")

    split = split_indices(question_rows, config)
    for index_value, row in enumerate(question_rows):
        row["split"] = split[index_value]

    by_split = {
        name: [row for row in question_rows if row["split"] == name]
        for name in ("train", "dev", "test")
    }
    best_router, tuning_rows = tune_router(
        [row["question"] for row in by_split["train"]],
        [row["silver_label"] for row in by_split["train"]],
        [row["question"] for row in by_split["dev"]],
        [row["silver_label"] for row in by_split["dev"]],
        config["seed"],
    )
    router = build_router(
        best_router["feature_kind"],
        best_router["class_weight"],
        best_router["c_value"],
        config["seed"],
    )
    train_dev = by_split["train"] + by_split["dev"]
    router.fit(
        [row["question"] for row in train_dev],
        [row["silver_label"] for row in train_dev],
    )
    test_questions = [row["question"] for row in by_split["test"]]
    test_labels = [row["silver_label"] for row in by_split["test"]]
    test_ids = [row["id"] for row in by_split["test"]]
    predictions = router.predict(test_questions).tolist()
    probabilities = router.predict_proba(test_questions)
    classes = router.named_steps["classifier"].classes_.tolist()
    confusion = confusion_matrix(test_labels, predictions, labels=list(METHODS))

    test_predictions = []
    for row, predicted, probability in zip(by_split["test"], predictions, probabilities):
        test_predictions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "question_type": row["question_type"],
                "silver_label": row["silver_label"],
                "predicted_label": predicted,
                **{f"p_{name}": float(value) for name, value in zip(classes, probability)},
            }
        )

    test_policies: dict[str, dict] = {}
    for method in METHODS:
        test_policies[method] = evaluate_policy(
            question_rows,
            backend_rows,
            test_ids,
            [method] * len(test_ids),
        )
    test_policies["adaptive"] = evaluate_policy(
        question_rows, backend_rows, test_ids, predictions
    )
    test_policies["oracle"] = evaluate_policy(
        question_rows, backend_rows, test_ids, test_labels
    )

    test_policies_by_type: dict[str, dict[str, dict]] = {}
    for question_type in config["question_types"]:
        positions = [
            position
            for position, row in enumerate(by_split["test"])
            if row["question_type"] == question_type
        ]
        ids = [test_ids[position] for position in positions]
        test_policies_by_type[question_type] = {}
        for method in METHODS:
            test_policies_by_type[question_type][method] = evaluate_policy(
                question_rows, backend_rows, ids, [method] * len(ids)
            )
        test_policies_by_type[question_type]["adaptive"] = evaluate_policy(
            question_rows,
            backend_rows,
            ids,
            [predictions[position] for position in positions],
        )
        test_policies_by_type[question_type]["oracle"] = evaluate_policy(
            question_rows,
            backend_rows,
            ids,
            [test_labels[position] for position in positions],
        )

    # Adaptive-RAG also mixes silver labels with dataset-induced labels. We test
    # that idea as an ablation, but select the production router on dev Macro-F1.
    prior_ablation = []
    prior_label = {
        "Fact Retrieval": "vector",
        "Complex Reasoning": "path_proxy",
    }
    for repeats in (1, 2):
        augmented_questions = [row["question"] for row in by_split["train"]]
        augmented_labels = [row["silver_label"] for row in by_split["train"]]
        for _ in range(repeats):
            augmented_questions.extend(row["question"] for row in by_split["train"])
            augmented_labels.extend(prior_label[row["question_type"]] for row in by_split["train"])
        ablation_router = build_router(
            best_router["feature_kind"],
            best_router["class_weight"],
            best_router["c_value"],
            config["seed"],
        )
        ablation_router.fit(augmented_questions, augmented_labels)
        ablation_predictions = ablation_router.predict(
            [row["question"] for row in by_split["dev"]]
        )
        prior_ablation.append(
            {
                "prior_repeats": repeats,
                "dev_accuracy": float(
                    accuracy_score(
                        [row["silver_label"] for row in by_split["dev"]],
                        ablation_predictions,
                    )
                ),
                "dev_macro_f1": float(
                    f1_score(
                        [row["silver_label"] for row in by_split["dev"]],
                        ablation_predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    majority_label = Counter(row["silver_label"] for row in train_dev).most_common(1)[0][0]
    summary = {
        "protocol_status": (
            "Offline retrieval-layer proxy run. These numbers are not official LightRAG/PathRAG "
            "generation results; official backends require an LLM API or Ollama."
        ),
        "benchmark": {
            "name": "GraphRAG-Bench Medical",
            "commit": benchmark_commit(args.benchmark_dir),
            "corpus_words": len(corpus.split()),
            "chunks": len(chunks),
            "questions": len(question_rows),
            "question_type_counts": dict(Counter(row["question_type"] for row in question_rows)),
            "split_counts": dict(Counter(row["split"] for row in question_rows)),
        },
        "silver_label_distribution": dict(Counter(row["silver_label"] for row in question_rows)),
        "router": {
            "family": "TF-IDF + LogisticRegression",
            "selected_hyperparameters": best_router,
            "test_accuracy": float(accuracy_score(test_labels, predictions)),
            "test_macro_f1": float(
                f1_score(test_labels, predictions, average="macro", zero_division=0)
            ),
            "majority_label": majority_label,
            "majority_accuracy": float(np.mean([label == majority_label for label in test_labels])),
            "majority_macro_f1": float(
                f1_score(
                    test_labels,
                    [majority_label] * len(test_labels),
                    average="macro",
                    zero_division=0,
                )
            ),
            "labels": list(METHODS),
            "confusion_matrix": confusion.tolist(),
            "dataset_prior_ablation": prior_ablation,
        },
        "test_policies": test_policies,
        "test_policies_by_question_type": test_policies_by_type,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Retrieval proxies use TF-IDF and a chunk graph; they are not paper-faithful LLM graph extraction.",
            "Evidence coverage is a deterministic lexical proxy, while the official benchmark uses an LLM judge.",
            "The extractive answer proxy is reported only as a pipeline sanity check, not as an LLM answer score.",
        ],
    }

    pd.DataFrame(flat_backend_rows).to_csv(args.output_dir / "backend_results.csv", index=False)
    pd.DataFrame(test_predictions).to_csv(args.output_dir / "router_test_predictions.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(args.output_dir / "router_tuning.csv", index=False)
    with (args.output_dir / "processed_questions.jsonl").open("w", encoding="utf-8") as handle:
        for row in question_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    question_lookup = {row["id"]: row for row in question_rows}
    chosen_cases: list[tuple[str, str, str]] = []
    for method in METHODS:
        match = next(
            (
                (question_id, truth, predicted)
                for question_id, truth, predicted in zip(test_ids, test_labels, predictions)
                if truth == method and predicted == truth
            ),
            None,
        )
        if match:
            chosen_cases.append(match)
    for question_type in config["question_types"]:
        mismatch = next(
            (
                (question_id, truth, predicted)
                for question_id, truth, predicted in zip(test_ids, test_labels, predictions)
                if question_lookup[question_id]["question_type"] == question_type
                and truth != predicted
            ),
            None,
        )
        if mismatch:
            chosen_cases.append(mismatch)
    representative_cases = []
    for question_id, truth, predicted in chosen_cases[:5]:
        question = question_lookup[question_id]
        methods = {}
        for method in METHODS:
            retrieval = index.retrieve(question["question"], method)
            methods[method] = {
                "metrics": backend_rows[question_id][method],
                "contexts": retrieval.contexts,
                "trace": retrieval.trace,
            }
        representative_cases.append(
            {
                **question,
                "oracle_route": truth,
                "predicted_route": predicted,
                "route_correct": truth == predicted,
                "methods": methods,
            }
        )
    (args.output_dir / "representative_cases.json").write_text(
        json.dumps(representative_cases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(router, args.output_dir.parent / "artifacts" / "router.joblib")
    save_plots(summary, confusion, summary["silver_label_distribution"], args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
