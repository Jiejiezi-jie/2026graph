from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Frozen phase-one proxy: keep it runnable as a plain script from any cwd.
# This directory provides `backends`; the repository root provides `src.*`.
CODE_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE_DIR.parents[2]), str(CODE_DIR)]

import joblib

from backends import OfflineGraphIndex
from src.backend.common.data import chunk_by_word_window, load_medical_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one routed query")
    parser.add_argument("question")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.json"))
    parser.add_argument("--router", type=Path, default=Path("artifacts/router.joblib"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus, _ = load_medical_benchmark(args.benchmark_dir)
    chunks = chunk_by_word_window(
        corpus, config["chunk_words"], config["chunk_overlap_words"]
    )
    index = OfflineGraphIndex(
        chunks,
        vector_top_k=config["vector_top_k"],
        light_top_k=config["light_top_k"],
        path_top_k=config["path_top_k"],
        graph_neighbors=config["graph_neighbors"],
        graph_similarity_floor=config["graph_similarity_floor"],
        adjacent_edge_weight=config["adjacent_edge_weight"],
    )
    router = joblib.load(args.router)
    route = str(router.predict([args.question])[0])
    probabilities = router.predict_proba([args.question])[0]
    classes = router.named_steps["classifier"].classes_
    result = index.retrieve(args.question, route)
    print(
        json.dumps(
            {
                "question": args.question,
                "selected_route": route,
                "route_probabilities": {
                    str(name): float(value) for name, value in zip(classes, probabilities)
                },
                "answer_proxy": result.answer_proxy,
                "context_words": result.context_words,
                "latency_ms": result.latency_ms,
                "retrieved_chunk_ids": result.chunk_ids,
                "trace": result.trace,
                "context_previews": [context[:350] for context in result.contexts],
                "warning": "Extractive offline proxy, not an LLM-generated medical answer.",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

