from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend.common.run_official_experiment import _check_cuda_headroom, _prepare_resume_output
from src.backend.common.data import load_medical_benchmark
from src.backend.lightRAG.lightrag_backend import LightRAGBackend
from src.backend.common.model_client import build_chat_client, build_embedding_client
from src.backend.pathRAG.pathrag_backend import PathRAGBackend
from src.backend.vector.vector_backend import VectorRAGBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run current backends on the frozen router expansion set."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/router_data_expansion.json")
    )
    parser.add_argument(
        "--backend", choices=["vector", "lightrag", "pathrag"], required=True
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


def build_backend(
    name: str, config: dict[str, Any], llm: Any, embedding: Any
) -> Any:
    common = {
        "llm": llm,
        "embedding": embedding,
        "chunk_tokens": config["chunk_tokens"],
        "chunk_overlap_tokens": config["chunk_overlap_tokens"],
        "max_context_tokens": config["max_context_tokens"],
        "generation_prompt": config["generation_prompt"],
    }
    working_dir = resolve(config["index_dirs"][name])
    if name == "vector":
        return VectorRAGBackend(
            working_dir=working_dir, top_k=config["vector_top_k"], **common
        )
    if name == "lightrag":
        return LightRAGBackend(
            working_dir=working_dir,
            top_k=config["lightrag_top_k"],
            chunk_top_k=config["lightrag_chunk_top_k"],
            max_entity_tokens=config["lightrag_max_entity_tokens"],
            max_relation_tokens=config["lightrag_max_relation_tokens"],
            max_total_tokens=config["lightrag_max_total_tokens"],
            **common,
        )
    return PathRAGBackend(
        upstream_dir=ROOT / "deps" / "PathRAG",
        working_dir=working_dir,
        top_k=config["pathrag_top_k"],
        **common,
    )


def validate_frozen_inputs(config: dict[str, Any], questions: list[dict[str, Any]]) -> None:
    manifest_path = resolve(config["selection_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(resolve(config["question_file"])) != manifest["files"]["question_file_sha256"]:
        raise ValueError("Expansion question file no longer matches its frozen manifest")
    expected = set(manifest["expansion"]["question_ids"])
    actual = {row["id"] for row in questions}
    if len(questions) != 180 or actual != expected:
        raise ValueError("Expansion question IDs do not match the frozen 180-question set")
    if any(row["split"] == "test" for row in questions):
        raise ValueError("Expansion is forbidden from adding test questions")

    light_graph = resolve(config["index_dirs"]["lightrag"]) / "medical" / "graph_chunk_entity_relation.graphml"
    path_graph = resolve(config["index_dirs"]["pathrag"]) / "graph_chunk_entity_relation.graphml"
    if sha256(light_graph) != sha256(path_graph):
        raise ValueError("LightRAG and PathRAG must use the byte-identical shared graph")


async def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _check_cuda_headroom(config)
    questions = load_jsonl(resolve(config["question_file"]))
    validate_frozen_inputs(config, questions)
    corpus, _ = load_medical_benchmark(resolve(config["benchmark_dir"]))

    llm = build_chat_client(config["llm"])
    embedding = build_embedding_client(config["embedding"])
    backend = build_backend(args.backend, config, llm, embedding)
    output_dir = resolve(config["results_dir"]) / "p1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.backend}.jsonl"
    completed = _prepare_resume_output(output_path)
    try:
        if isinstance(backend, VectorRAGBackend):
            backend.load_cached_index(corpus)
        with output_path.open("a", encoding="utf-8") as handle:
            for position, question in enumerate(questions, start=1):
                if question["id"] in completed:
                    continue
                print(
                    f"[{args.backend}] {position}/{len(questions)} {question['id']}",
                    flush=True,
                )
                try:
                    row = await backend.query(question["question"], question["id"])
                    row.update(
                        {
                            "question": question["question"],
                            "question_type": question["question_type"],
                            "ground_truth": question["answer"],
                            "evidence": question["evidence"],
                            "evidence_relations": question.get("evidence_relations", ""),
                            "split": question["split"],
                            "error": None,
                            "expansion_set": True,
                        }
                    )
                    if not row["answer"].strip() or not row["contexts"]:
                        raise RuntimeError("Backend returned an empty answer or no contexts")
                except Exception as exc:
                    row = {
                        "question_id": question["id"],
                        "method": args.backend,
                        "question": question["question"],
                        "question_type": question["question_type"],
                        "ground_truth": question["answer"],
                        "evidence": question["evidence"],
                        "evidence_relations": question.get("evidence_relations", ""),
                        "split": question["split"],
                        "expansion_set": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
    finally:
        await backend.close()
        llm.unload()
        embedding.unload()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
