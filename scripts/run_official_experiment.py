from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from src.data import load_medical_benchmark
from src.official_backends.lightrag_backend import LightRAGBackend
from src.official_backends.model_client import (
    TransformersChatClient,
    TransformersEmbeddingClient,
)
from src.official_backends.pathrag_backend import PathRAGBackend
from src.official_backends.vector_backend import VectorRAGBackend
from src.official_data import p0_subset, stratified_sample_and_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aligned official RAG backends")
    parser.add_argument("--config", type=Path, default=Path("configs/official_local.json"))
    parser.add_argument("--stage", choices=["p0", "p1"], default="p0")
    parser.add_argument(
        "--backend", choices=["all", "vector", "lightrag", "pathrag"], default="all"
    )
    parser.add_argument("--skip-index", action="store_true")
    return parser.parse_args()


def _resolve(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(json.loads(line)["question_id"])
    return completed


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _check_cuda_headroom(config: dict[str, Any]) -> None:
    cuda_entries = [config["llm"], config["embedding"]]
    if not any(str(entry.get("device", "")).startswith("cuda:") for entry in cuda_entries):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but PyTorch cannot initialize CUDA")
    for entry in cuda_entries:
        device = str(entry.get("device", ""))
        if not device.startswith("cuda:"):
            continue
        free, total = torch.cuda.mem_get_info(device)
        free_gib = free / 2**30
        required_gib = float(entry.get("min_free_gib", 0))
        print(
            f"[preflight] {device}: {free_gib:.2f}/{total / 2**30:.2f} GiB free; "
            f"requires {required_gib:.2f} GiB",
            flush=True,
        )
        if free_gib < required_gib:
            raise RuntimeError(
                f"Refusing shared-GPU run: {device} has {free_gib:.2f} GiB free, "
                f"below configured minimum {required_gib:.2f} GiB"
            )


def _build_backends(
    names: list[str],
    project_dir: Path,
    config: dict[str, Any],
    llm: TransformersChatClient,
    embedding: TransformersEmbeddingClient,
) -> dict[str, Any]:
    results_dir = _resolve(project_dir, config["results_dir"])
    common = {
        "llm": llm,
        "embedding": embedding,
        "chunk_tokens": config["chunk_tokens"],
        "chunk_overlap_tokens": config["chunk_overlap_tokens"],
        "max_context_tokens": config["max_context_tokens"],
        "generation_prompt": config["generation_prompt"],
    }
    backends: dict[str, Any] = {}
    if "vector" in names:
        backends["vector"] = VectorRAGBackend(
            working_dir=results_dir / "indexes" / "vector",
            top_k=config["vector_top_k"],
            **common,
        )
    if "lightrag" in names:
        backends["lightrag"] = LightRAGBackend(
            working_dir=results_dir / "indexes" / "lightrag",
            top_k=config["lightrag_top_k"],
            chunk_top_k=config["lightrag_chunk_top_k"],
            **common,
        )
    if "pathrag" in names:
        backends["pathrag"] = PathRAGBackend(
            upstream_dir=project_dir / "third_party" / "PathRAG",
            working_dir=results_dir / "indexes" / "pathrag",
            top_k=config["pathrag_top_k"],
            **common,
        )
    return backends


async def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    config_path = _resolve(project_dir, str(args.config))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _check_cuda_headroom(config)
    benchmark_dir = _resolve(project_dir, config["benchmark_dir"])
    results_dir = _resolve(project_dir, config["results_dir"])
    corpus, all_questions = load_medical_benchmark(benchmark_dir)
    p1_rows = stratified_sample_and_split(
        all_questions,
        config["question_types"],
        config["p1_per_type"],
        config["seed"],
    )
    questions = (
        p0_subset(p1_rows, config["question_types"], config["p0_per_type"])
        if args.stage == "p0"
        else p1_rows
    )
    split_dir = results_dir / "splits"
    _write_jsonl(split_dir / "p1_questions.jsonl", p1_rows)
    _write_jsonl(
        split_dir / "p0_questions.jsonl",
        p0_subset(p1_rows, config["question_types"], config["p0_per_type"]),
    )

    llm_config = config["llm"]
    embedding_config = config["embedding"]
    llm = TransformersChatClient(
        model_path=llm_config["model_path"],
        device=llm_config["device"],
        dtype=llm_config["dtype"],
        temperature=llm_config["temperature"],
        max_new_tokens=llm_config["max_new_tokens"],
        max_callback_new_tokens=llm_config["max_callback_new_tokens"],
        trust_remote_code=llm_config["trust_remote_code"],
    )
    embedding = TransformersEmbeddingClient(
        model_path=embedding_config["model_path"],
        device=embedding_config["device"],
        batch_size=embedding_config["batch_size"],
        max_length=embedding_config["max_length"],
        normalize=embedding_config["normalize"],
    )
    names = ["vector", "lightrag", "pathrag"] if args.backend == "all" else [args.backend]
    backends = _build_backends(names, project_dir, config, llm, embedding)
    stage_dir = results_dir / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    index_stats_path = stage_dir / "index_stats.json"
    index_stats: dict[str, Any] = _load_json_object(index_stats_path)
    try:
        for name, backend in backends.items():
            output_path = stage_dir / f"{name}.jsonl"
            completed = _load_completed(output_path)
            if not args.skip_index:
                print(f"[{name}] indexing or loading index", flush=True)
                index_stats[name] = await backend.index(corpus)
                index_stats_path.write_text(
                    json.dumps(index_stats, indent=2), encoding="utf-8"
                )
            with output_path.open("a", encoding="utf-8") as handle:
                for position, question in enumerate(questions, start=1):
                    if question["id"] in completed:
                        continue
                    print(
                        f"[{name}] {position}/{len(questions)} {question['id']}",
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
                            }
                        )
                        if not row["answer"].strip() or not row["contexts"]:
                            raise RuntimeError("Backend returned an empty answer or no contexts")
                    except Exception as exc:
                        row = {
                            "question_id": question["id"],
                            "method": name,
                            "question": question["question"],
                            "question_type": question["question_type"],
                            "ground_truth": question["answer"],
                            "evidence": question["evidence"],
                            "split": question["split"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        if args.stage == "p0":
                            raise
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
    finally:
        for backend in backends.values():
            await backend.close()
        llm.unload()
        embedding.unload()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
