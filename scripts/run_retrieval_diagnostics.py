#!/usr/bin/env python3
"""Run three shared-data retrievers over the 62 phase-one training queries."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase4_retrieval import (
    FORBIDDEN_RESULT_KEYS,
    Phase4Error,
    canonical_json,
    diagnostics_for_results,
    jsonl_dump,
    load_resources,
    load_train_questions,
    render_phase4_report,
    run_all_retrievers,
    sha256_file,
    structural_sha256,
    validate_result,
    write_text_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "retrieval_config.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    try:
        resources, config, paths, index_metadata = load_resources(ROOT, config_path)
        input_hashes_before = {
            key: sha256_file(paths[key])
            for key in (
                "chunks",
                "nodes",
                "edges",
                "chunk_entity_map",
                "train_questions",
            )
        }
        questions = load_train_questions(paths["train_questions"])
        results = run_all_retrievers(resources, questions)
        if len(results) != len(questions) * 3:
            raise Phase4Error("each training query must have three retrieval results")
        for result in results:
            validate_result(result, int(config["top_k_chunks"]))
            if FORBIDDEN_RESULT_KEYS & set(result):
                raise Phase4Error("supervision field leaked into retrieval output")
        result_counts = collections.Counter(row["retriever"] for row in results)
        if set(result_counts.values()) != {62}:
            raise Phase4Error(f"retriever result counts are invalid: {result_counts}")
        diagnostics = diagnostics_for_results(questions, results)
        write_text_atomic(paths["results"], jsonl_dump(results))
        write_text_atomic(paths["diagnostics"], canonical_json(diagnostics, pretty=True))
        manifest = {
            "schema_version": config["schema_version"],
            "scope": "coverage-only shared-data retrieval diagnostics",
            "implementation_boundary": (
                "course simplification inspired by LightRAG Local and PathRAG; "
                "not a complete reproduction"
            ),
            "random_seed": config["random_seed"],
            "config_file": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(config_path),
            "inputs": {
                key: {
                    "path": paths[key].relative_to(ROOT).as_posix(),
                    "sha256": value,
                }
                for key, value in input_hashes_before.items()
            },
            "question_count": len(questions),
            "question_type_counts": dict(
                sorted(collections.Counter(row["question_type"] for row in questions).items())
            ),
            "result_count": len(results),
            "retriever_result_counts": dict(sorted(result_counts.items())),
            "embedding": index_metadata["embedding"],
            "index_metadata_file": paths["index_metadata"].relative_to(ROOT).as_posix(),
            "index_metadata_sha256": sha256_file(paths["index_metadata"]),
            "index_files": index_metadata["index_files"],
            "retrieval_parameters": {
                key: config[key]
                for key in (
                    "top_k_chunks",
                    "max_seed_entities",
                    "entity_similarity_threshold",
                    "neighborhood_hops",
                    "max_path_hops",
                    "max_paths_per_pair",
                    "excluded_path_relations",
                    "neighborhood_scoring",
                    "path_scoring",
                )
            },
            "result_structural_sha256": structural_sha256(results),
            "diagnostics_structural_sha256": structural_sha256(diagnostics),
            "forbidden_supervision_fields_written": [],
            "generative_llm_calls": 0,
            "answer_generation_performed": False,
            "vector_fallback_for_graph_retrievers": False,
        }
        write_text_atomic(paths["manifest"], canonical_json(manifest, pretty=True))
        write_text_atomic(
            paths["report"], render_phase4_report(diagnostics, manifest, config)
        )
        input_hashes_after = {
            key: sha256_file(paths[key]) for key in input_hashes_before
        }
        if input_hashes_before != input_hashes_after:
            raise Phase4Error("a phase-one, phase-two, or phase-three input changed")
    except (Phase4Error, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"阶段4诊断失败：{exc}", file=sys.stderr)
        return 1
    print(canonical_json(manifest, pretty=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
