"""Phase-five grounded generation, official evaluation, and pseudo-labeling.

The generation side projects only question IDs and question text. Reference
answers and reference evidence are loaded only by the separate evaluation
entry point after all 186 generation rows exist.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import math
import os
import re
import statistics
import string
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.phase4_retrieval import BGEEncoder


RETRIEVERS = ("vector", "neighborhood", "path")
GENERATION_PAYLOAD_FIELDS = {"answer", "citations", "insufficient_evidence"}
SUPERVISION_FIELDS = {"evidence", "evidence_triple", "question_type"}
GENERATION_CACHE_VERSION = "phase5-generation-cache-v1"
EVALUATION_CACHE_VERSION = "phase5-evaluation-cache-v1"


class Phase5Error(RuntimeError):
    """Raised when a phase-five protocol or artifact invariant fails."""


class PermanentLLMError(Phase5Error):
    """Raised for authentication, request-shape, or unknown-model failures."""


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def jsonl_dump(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except FileNotFoundError as exc:
        raise Phase5Error(f"missing required file: {path}") from exc


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase5Error(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase5Error(f"invalid JSON file {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise Phase5Error(f"missing JSONL file: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Phase5Error(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise Phase5Error(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def load_config(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = read_json(config_path)
    fixed = {
        "random_seed": 42,
        "max_context_chunks": 5,
        "expected_effective_generation_calls": 176,
    }
    if any(config.get(key) != value for key, value in fixed.items()):
        raise Phase5Error("phase-five fixed top-level configuration mismatch")
    if config.get("retrievers") != list(RETRIEVERS):
        raise Phase5Error("retriever order must be vector, neighborhood, path")
    generation = config.get("generation", {})
    expected_generation = {
        "temperature": 0,
        "thinking": "disabled",
        "max_retries": 2,
        "response_format": "json_object",
    }
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        raise Phase5Error("fixed generation parameters were changed")
    evaluation = config.get("evaluation", {})
    if evaluation.get("answer_correctness_implementation") != "graphrag_bench_official":
        raise Phase5Error("GraphRAG-Bench official correctness is required")
    if not math.isclose(float(evaluation.get("factuality_weight", -1)), 0.75):
        raise Phase5Error("factuality weight must be 0.75")
    if not math.isclose(float(evaluation.get("semantic_weight", -1)), 0.25):
        raise Phase5Error("semantic weight must be 0.25")
    labeling = config.get("labeling", {})
    if (
        labeling.get("answer_correctness_threshold") != 0.75
        or labeling.get("evidence_coverage_threshold") != 0.5
        or labeling.get("difficulty_order")
        != {"vector": 0, "neighborhood": 1, "path": 2}
    ):
        raise Phase5Error("pseudo-label thresholds or difficulty order changed")

    paths: dict[str, Path] = {}
    for key, item in config["inputs"].items():
        paths[key] = (root / item["path"]).resolve()
    paths["system_prompt"] = (root / generation["system_prompt"]).resolve()
    paths["benchmark_directory"] = (
        root / evaluation["benchmark_directory"]
    ).resolve()
    for key, value in config["cache"].items():
        paths[key] = (root / value).resolve()
    output_dir = (root / config["outputs"]["directory"]).resolve()
    paths["output_directory"] = output_dir
    for key in ("train_answers", "method_scores", "router_labels", "method_summary", "manifest"):
        paths[key] = output_dir / config["outputs"][key]
    paths["report"] = (root / config["outputs"]["report"]).resolve()
    paths["config"] = config_path.resolve()
    return config, paths


def validate_frozen_inputs(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, item in config["inputs"].items():
        actual = sha256_file(paths[key])
        expected = item["sha256"]
        if actual != expected:
            raise Phase5Error(
                f"frozen input changed for {key}: expected {expected}, got {actual}"
            )
        hashes[key] = actual
    return hashes


def validate_training_schema(path: Path, expected_count: int) -> None:
    rows = read_jsonl(path)
    if len(rows) != expected_count:
        raise Phase5Error(f"expected {expected_count} training questions, got {len(rows)}")
    if len({row.get("id") for row in rows}) != expected_count:
        raise Phase5Error("training question IDs are not unique")
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise Phase5Error("training question is empty")
        if not isinstance(row.get("answer"), (str, list)) or not row["answer"]:
            raise Phase5Error("training data has no usable standard answer")
        if "evidence" not in row or not row["evidence"]:
            raise Phase5Error(
                "training data has no standard evidence; stopping before API calls"
            )


def load_generation_questions(path: Path) -> list[dict[str, str]]:
    """Retain only ID and question; supervision is discarded immediately."""
    projected = []
    for row in read_jsonl(path):
        projected.append({"id": str(row["id"]), "question": str(row["question"])})
    return projected


def load_evaluation_references(path: Path) -> dict[str, dict[str, Any]]:
    """Load supervision only from the evaluation entry point after generation."""
    references: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        references[str(row["id"])] = {
            "answers": parse_reference_answers(row["answer"]),
            "evidence": parse_reference_evidence(row["evidence"]),
        }
    return references


def load_chunks(path: Path, expected_count: int) -> tuple[list[dict[str, str]], dict[str, str]]:
    chunks = read_jsonl(path)
    if len(chunks) != expected_count:
        raise Phase5Error(f"expected {expected_count} chunks, got {len(chunks)}")
    projected = []
    by_id: dict[str, str] = {}
    for row in chunks:
        chunk_id, text = row.get("chunk_id"), row.get("text")
        if not isinstance(chunk_id, str) or not isinstance(text, str) or not text.strip():
            raise Phase5Error("invalid phase-two chunk")
        if chunk_id in by_id:
            raise Phase5Error(f"duplicate chunk ID: {chunk_id}")
        by_id[chunk_id] = text
        projected.append({"chunk_id": chunk_id, "text": text})
    return projected, by_id


def load_and_validate_retrieval_results(
    path: Path,
    questions: Sequence[Mapping[str, str]],
    chunk_by_id: Mapping[str, str],
    expected_count: int,
    max_chunks: int,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected_count:
        raise Phase5Error(f"expected {expected_count} retrieval rows, got {len(rows)}")
    question_by_id = {row["id"]: row["question"] for row in questions}
    observed: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in rows:
        query_id, retriever = row.get("query_id"), row.get("retriever")
        if query_id not in question_by_id or retriever not in RETRIEVERS:
            raise Phase5Error("retrieval result has unknown query or retriever")
        if row.get("query") != question_by_id[query_id]:
            raise Phase5Error(f"retrieval query text mismatch for {query_id}")
        observed[(query_id, retriever)] += 1
        retrieved = row.get("retrieved_chunks")
        if not isinstance(retrieved, list) or len(retrieved) > max_chunks:
            raise Phase5Error("retrieval result violates the five-chunk budget")
        ids = [item.get("chunk_id") for item in retrieved]
        if any(chunk_id not in chunk_by_id for chunk_id in ids):
            raise Phase5Error("retrieval result references an unknown chunk")
        if len(ids) != len(set(ids)):
            raise Phase5Error("retrieval result repeats a chunk")
        if [item.get("rank") for item in retrieved] != list(range(1, len(ids) + 1)):
            raise Phase5Error("retrieval ranks are not consecutive")
    expected_pairs = {
        (query_id, retriever) for query_id in question_by_id for retriever in RETRIEVERS
    }
    if set(observed) != expected_pairs or any(value != 1 for value in observed.values()):
        raise Phase5Error("retrieval results are not a complete 62 x 3 matrix")
    return rows


def runtime_from_env(*, online: bool, cache_dir: Path | None = None) -> dict[str, str]:
    values = {
        name: os.environ.get(name, "").strip()
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    }
    if online:
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise Phase5Error(f"missing LLM environment variables: {', '.join(missing)}")
    elif not values["LLM_MODEL"] and cache_dir is not None:
        models = {
            str(item.get("model"))
            for path in cache_dir.glob("*.json")
            if isinstance((item := read_json(path)), dict) and item.get("model")
        }
        if len(models) == 1:
            values["LLM_MODEL"] = next(iter(models))
        elif not models:
            raise Phase5Error("offline replay has no caches from which to infer LLM_MODEL")
        else:
            raise Phase5Error("offline replay caches contain multiple model names")
    return values


def graph_context_from_result(result: Mapping[str, Any]) -> str:
    if result["retriever"] == "vector":
        return ""
    graph_context = {
        "linked_entities": result.get("linked_entities", []),
        "retrieved_nodes": result.get("retrieved_nodes", []),
        "retrieved_edges": result.get("retrieved_edges", []),
        "paths": result.get("paths", []) if result["retriever"] == "path" else [],
    }
    return "GRAPH CONTEXT (retriever output only):\n" + canonical_json(
        graph_context, pretty=True
    ).rstrip()


def build_generation_messages(
    system_prompt: str,
    question: str,
    result: Mapping[str, Any],
    chunk_by_id: Mapping[str, str],
    max_chunks: int,
) -> tuple[list[dict[str, str]], list[str], str]:
    retrieved = sorted(result["retrieved_chunks"], key=lambda row: row["rank"])[
        :max_chunks
    ]
    chunk_ids = [row["chunk_id"] for row in retrieved]
    evidence_blocks = [
        f"[CHUNK_ID: {chunk_id}]\n{chunk_by_id[chunk_id]}" for chunk_id in chunk_ids
    ]
    sections = ["EVIDENCE CHUNKS:", "\n\n".join(evidence_blocks)]
    graph_context = graph_context_from_result(result)
    if graph_context:
        sections.extend(["", graph_context])
    context = "\n".join(sections)
    user_prompt = f"QUESTION:\n{question}\n\n{context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages, chunk_ids, context


def validate_generation_payload(
    payload: Any, allowed_chunk_ids: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict) or set(payload) != GENERATION_PAYLOAD_FIELDS:
        raise Phase5Error(
            "generation payload must contain exactly answer, citations, insufficient_evidence"
        )
    answer = payload["answer"]
    citations = payload["citations"]
    insufficient = payload["insufficient_evidence"]
    if not isinstance(answer, str) or not answer.strip():
        raise Phase5Error("generation answer must be a non-empty string")
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        raise Phase5Error("generation citations must be a string list")
    if not isinstance(insufficient, bool):
        raise Phase5Error("insufficient_evidence must be boolean")
    if insufficient and answer.strip() != "INSUFFICIENT_EVIDENCE":
        raise Phase5Error("insufficient answer must be exactly INSUFFICIENT_EVIDENCE")
    if not insufficient and answer.strip() == "INSUFFICIENT_EVIDENCE":
        raise Phase5Error("INSUFFICIENT_EVIDENCE requires insufficient_evidence=true")
    deduplicated = list(dict.fromkeys(citations))
    allowed = set(allowed_chunk_ids)
    invalid = [item for item in deduplicated if item not in allowed]
    return {
        "answer": answer.strip(),
        "citations": deduplicated,
        "insufficient_evidence": insufficient,
    }, invalid


def usage_from_response(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    return {
        key: int(usage.get(key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


class OpenAICompatibleClient:
    def __init__(self, runtime: Mapping[str, str], generation_config: Mapping[str, Any]):
        self.api_key = runtime["LLM_API_KEY"]
        self.base_url = runtime["LLM_BASE_URL"].rstrip("/")
        self.model = runtime["LLM_MODEL"]
        self.config = dict(generation_config)

    def call_once(
        self, messages: Sequence[Mapping[str, str]], *, strict_object: bool
    ) -> tuple[dict[str, Any], float]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
            "thinking": {"type": self.config["thinking"]},
            "stream": False,
        }
        if strict_object:
            body["response_format"] = {"type": self.config["response_format"]}
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=canonical_json(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "2026graph-phase5/1.0",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                request, timeout=float(self.config["timeout_seconds"])
            ) as reply:
                response_bytes = reply.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            error_class = PermanentLLMError if exc.code in {400, 401, 403, 404} else Phase5Error
            raise error_class(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise Phase5Error(f"LLM network error: {exc.reason}") from exc
        elapsed = time.monotonic() - started
        try:
            response = json.loads(response_bytes)
        except json.JSONDecodeError as exc:
            raise Phase5Error("LLM response body is not JSON") from exc
        return response, elapsed


def response_content(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise Phase5Error("LLM response is missing choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise Phase5Error("LLM returned empty content")
    return content.strip()


def generation_cache_path(
    cache_dir: Path, query_id: str, retriever: str, prompt_hash: str
) -> Path:
    safe_query = re.sub(r"[^A-Za-z0-9_-]+", "_", query_id).strip("_")
    if not safe_query or retriever not in RETRIEVERS:
        raise Phase5Error("unsafe generation cache identity")
    return cache_dir / f"{safe_query}__{retriever}__{prompt_hash[:20]}.json"


def generation_request_fingerprint(
    prompt_hash: str, model: str, generation_config: Mapping[str, Any]
) -> str:
    fixed = {
        key: generation_config[key]
        for key in ("temperature", "thinking", "max_tokens", "response_format")
    }
    return sha256_text(canonical_json({"prompt_sha256": prompt_hash, "model": model, **fixed}))


def validate_generation_cache(
    cache: Any,
    *,
    query_id: str,
    retriever: str,
    prompt_hash: str,
    request_fingerprint: str,
    model: str,
    allowed_chunk_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(cache, dict):
        raise Phase5Error("generation cache is not an object")
    if (
        cache.get("cache_version") != GENERATION_CACHE_VERSION
        or cache.get("query_id") != query_id
        or cache.get("retriever") != retriever
        or cache.get("prompt_sha256") != prompt_hash
        or cache.get("request_fingerprint") != request_fingerprint
        or cache.get("model") != model
        or cache.get("context_chunk_ids") != list(allowed_chunk_ids)
        or sha256_text(canonical_json(cache.get("request_messages"))) != prompt_hash
    ):
        raise Phase5Error("generation cache fingerprint mismatch")
    if cache.get("outcome") == "failure":
        return cache
    if cache.get("outcome") != "success":
        raise Phase5Error("generation cache has an invalid outcome")
    validated, invalid = validate_generation_payload(
        cache.get("validated_output"), allowed_chunk_ids
    )
    result = dict(cache)
    result["validated_output"] = validated
    result["invalid_citations"] = invalid
    return result


def call_generation_with_cache(
    *,
    client: OpenAICompatibleClient | None,
    messages: Sequence[Mapping[str, str]],
    query_id: str,
    retriever: str,
    chunk_ids: Sequence[str],
    cache_dir: Path,
    generation_config: Mapping[str, Any],
    model: str,
    offline: bool,
    retry_failures: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    prompt_hash = sha256_text(canonical_json(list(messages)))
    fingerprint = generation_request_fingerprint(prompt_hash, model, generation_config)
    cache_path = generation_cache_path(cache_dir, query_id, retriever, prompt_hash)
    if cache_path.exists():
        try:
            cache = validate_generation_cache(
                read_json(cache_path),
                query_id=query_id,
                retriever=retriever,
                prompt_hash=prompt_hash,
                request_fingerprint=fingerprint,
                model=model,
                allowed_chunk_ids=chunk_ids,
            )
        except Phase5Error:
            cache = None
        if cache is not None and (cache["outcome"] != "failure" or not retry_failures):
            return cache, {"api_calls": 0, "cache_hits": 1, "successful_calls": 0}
    if offline:
        raise Phase5Error(f"offline replay lacks valid generation cache for {query_id}/{retriever}")
    if client is None:
        raise Phase5Error("online generation client was not initialized")

    attempts: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    successful_cache: dict[str, Any] | None = None
    for attempt in range(int(generation_config["max_retries"]) + 1):
        try:
            response, elapsed = client.call_once(messages, strict_object=True)
            raw_responses.append(response)
            content = response_content(response)
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise Phase5Error(f"generation content is not strict JSON: {exc}") from exc
            validated, invalid = validate_generation_payload(payload, chunk_ids)
            usage = usage_from_response(response)
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "status": "success",
                    "elapsed_ms": round(elapsed * 1000, 6),
                    "token_usage": usage,
                    "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                }
            )
            successful_cache = {
                "cache_version": GENERATION_CACHE_VERSION,
                "query_id": query_id,
                "retriever": retriever,
                "prompt_sha256": prompt_hash,
                "request_fingerprint": fingerprint,
                "model": model,
                "context_chunk_ids": list(chunk_ids),
                "request_messages": list(messages),
                "outcome": "success",
                "validated_output": validated,
                "invalid_citations": invalid,
                "request_audit": {
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "elapsed_ms": round(sum(row["elapsed_ms"] for row in attempts), 6),
                    "token_usage": usage,
                },
                "raw_response": response,
            }
            break
        except Exception as exc:
            if isinstance(exc, PermanentLLMError):
                raise
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:800],
                }
            )
            if attempt < int(generation_config["max_retries"]):
                time.sleep(float(generation_config["retry_backoff_seconds"]) * (attempt + 1))
    if successful_cache is None:
        successful_cache = {
            "cache_version": GENERATION_CACHE_VERSION,
            "query_id": query_id,
            "retriever": retriever,
            "prompt_sha256": prompt_hash,
            "request_fingerprint": fingerprint,
            "model": model,
            "context_chunk_ids": list(chunk_ids),
            "request_messages": list(messages),
            "outcome": "failure",
            "failure_reason": attempts[-1].get("error", "unknown generation failure"),
            "request_audit": {
                "attempt_count": len(attempts),
                "attempts": attempts,
                "elapsed_ms": round(
                    sum(float(row.get("elapsed_ms", 0.0)) for row in attempts), 6
                ),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
            "raw_responses": raw_responses,
        }
    write_text_atomic(cache_path, canonical_json(successful_cache, pretty=True))
    return successful_cache, {
        "api_calls": len(attempts),
        "cache_hits": 0,
        "successful_calls": int(successful_cache["outcome"] == "success"),
    }


def direct_insufficient_row(result: Mapping[str, Any], model: str) -> dict[str, Any]:
    return {
        "schema_version": "phase5-v1",
        "query_id": result["query_id"],
        "query": result["query"],
        "retriever": result["retriever"],
        "retrieval_status": result["status"],
        "context_chunk_ids": [],
        "context_character_count": 0,
        "prompt_sha256": None,
        "generation_status": "retrieval_failure",
        "answer": "INSUFFICIENT_EVIDENCE",
        "citations": [],
        "insufficient_evidence": True,
        "invalid_citations": [],
        "retrieval_latency_ms": float(result.get("diagnostics", {}).get("latency_ms", 0.0)),
        "generation": {
            "model": model,
            "temperature": 0,
            "thinking": "disabled",
            "attempt_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "elapsed_ms": 0.0,
            "llm_called": False,
        },
        "cache": {"cache_key": None, "cache_file": None},
    }


def generation_row_from_cache(
    result: Mapping[str, Any],
    cache: Mapping[str, Any],
    context_character_count: int,
    cache_path: Path,
    root: Path,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    audit = cache["request_audit"]
    usage = audit.get("token_usage", {})
    if cache["outcome"] == "success":
        output = cache["validated_output"]
        status = "model_insufficient" if output["insufficient_evidence"] else "success"
        answer = output["answer"]
        citations = output["citations"]
        insufficient = output["insufficient_evidence"]
        invalid = cache.get("invalid_citations", [])
    else:
        status = "generation_failure"
        answer = "INSUFFICIENT_EVIDENCE"
        citations = []
        insufficient = True
        invalid = []
    return {
        "schema_version": "phase5-v1",
        "query_id": result["query_id"],
        "query": result["query"],
        "retriever": result["retriever"],
        "retrieval_status": result["status"],
        "context_chunk_ids": list(cache["context_chunk_ids"]),
        "context_character_count": int(context_character_count),
        "prompt_sha256": cache["prompt_sha256"],
        "generation_status": status,
        "answer": answer,
        "citations": citations,
        "insufficient_evidence": insufficient,
        "invalid_citations": invalid,
        "retrieval_latency_ms": float(result.get("diagnostics", {}).get("latency_ms", 0.0)),
        "generation": {
            "model": cache["model"],
            "temperature": generation_config["temperature"],
            "thinking": generation_config["thinking"],
            "attempt_count": int(audit.get("attempt_count", 0)),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "elapsed_ms": float(audit.get("elapsed_ms", 0.0)),
            "llm_called": True,
        },
        "cache": {
            "cache_key": cache_path.stem,
            "cache_file": cache_path.relative_to(root).as_posix(),
        },
    }


def generate_train_answers(
    root: Path,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    offline: bool,
    retry_failures: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_hashes = validate_frozen_inputs(config, paths)
    validate_training_schema(
        paths["train_questions"], config["inputs"]["train_questions"]["expected_count"]
    )
    questions = load_generation_questions(paths["train_questions"])
    _chunks, chunk_by_id = load_chunks(
        paths["chunks"], config["inputs"]["chunks"]["expected_count"]
    )
    retrieval_rows = load_and_validate_retrieval_results(
        paths["retrieval_results"],
        questions,
        chunk_by_id,
        config["inputs"]["retrieval_results"]["expected_count"],
        config["max_context_chunks"],
    )
    system_prompt = paths["system_prompt"].read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise Phase5Error("generation system prompt is empty")
    cache_dir = paths["generation_responses"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    runtime = runtime_from_env(online=not offline, cache_dir=cache_dir)
    client = None if offline else OpenAICompatibleClient(runtime, config["generation"])
    model = runtime["LLM_MODEL"]

    rows: list[dict[str, Any]] = []
    stats = collections.Counter()
    eligible = 0
    for position, result in enumerate(retrieval_rows, 1):
        if result["status"] != "success" or not result["retrieved_chunks"]:
            rows.append(direct_insufficient_row(result, model))
            stats["retrieval_failures_skipped"] += 1
            print(
                f"[{position:03d}/186] {result['query_id']}/{result['retriever']}: retrieval {result['status']}",
                flush=True,
            )
            continue
        eligible += 1
        messages, chunk_ids, context = build_generation_messages(
            system_prompt,
            result["query"],
            result,
            chunk_by_id,
            int(config["max_context_chunks"]),
        )
        prompt_hash = sha256_text(canonical_json(messages))
        cache_path = generation_cache_path(
            cache_dir, result["query_id"], result["retriever"], prompt_hash
        )
        cache, call_stats = call_generation_with_cache(
            client=client,
            messages=messages,
            query_id=result["query_id"],
            retriever=result["retriever"],
            chunk_ids=chunk_ids,
            cache_dir=cache_dir,
            generation_config=config["generation"],
            model=model,
            offline=offline,
            retry_failures=retry_failures,
        )
        stats.update(call_stats)
        rows.append(
            generation_row_from_cache(
                result,
                cache,
                len(context),
                cache_path,
                root,
                config["generation"],
            )
        )
        source = "cache" if call_stats["cache_hits"] else "API"
        print(
            f"[{position:03d}/186] {result['query_id']}/{result['retriever']}: {source} {cache['outcome']}",
            flush=True,
        )
    if eligible != int(config["expected_effective_generation_calls"]):
        raise Phase5Error(
            f"expected 176 LLM-eligible rows after Path failures, got {eligible}"
        )
    if len(rows) != 186:
        raise Phase5Error("generation did not produce exactly 186 rows")
    write_text_atomic(paths["train_answers"], jsonl_dump(rows))
    audit = {
        "schema_version": "phase5-v1",
        "mode": "offline" if offline else "online",
        "model": model,
        "input_sha256": input_hashes,
        "eligible_generation_rows": eligible,
        "retrieval_failures_skipped": stats["retrieval_failures_skipped"],
        "api_calls_this_run": stats["api_calls"],
        "successful_api_calls_this_run": stats["successful_calls"],
        "cache_hits_this_run": stats["cache_hits"],
        "train_answers_sha256": sha256_file(paths["train_answers"]),
    }
    write_text_atomic(paths["generation_run_audit"], canonical_json(audit, pretty=True))
    if validate_frozen_inputs(config, paths) != input_hashes:
        raise Phase5Error("a frozen phase-one, phase-two, or phase-four input changed")
    return rows, audit


def parse_reference_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        answers = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        stripped = value.strip()
        answers = []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                answers = [str(item).strip() for item in parsed if str(item).strip()]
        if not answers:
            answers = [item.strip() for item in stripped.split(" || ") if item.strip()]
    else:
        answers = []
    if not answers:
        raise Phase5Error("standard answer is empty")
    return answers


def parse_reference_evidence(value: Any) -> list[str]:
    if isinstance(value, list):
        evidence = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        evidence = [item.strip() for item in value.split(";") if item.strip()]
    else:
        evidence = []
    if not evidence:
        raise Phase5Error("standard evidence is empty")
    return evidence


def normalize_for_evidence(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(" " if unicodedata.category(char).startswith("P") else char for char in value)
    return " ".join(value.split())


def token_recall(reference: str, candidate: str) -> float:
    reference_tokens = normalize_for_evidence(reference).split()
    candidate_counts = collections.Counter(normalize_for_evidence(candidate).split())
    if not reference_tokens:
        return 0.0
    reference_counts = collections.Counter(reference_tokens)
    overlap = sum(
        min(count, candidate_counts.get(token, 0)) for token, count in reference_counts.items()
    )
    return overlap / len(reference_tokens)


def evidence_coverage(
    reference_evidence: Sequence[str],
    context_chunk_ids: Sequence[str],
    chunk_by_id: Mapping[str, str],
    threshold: float,
) -> tuple[float, list[dict[str, Any]]]:
    details = []
    for evidence in reference_evidence:
        normalized_evidence = normalize_for_evidence(evidence)
        best_chunk: str | None = None
        best_recall = 0.0
        match_method: str | None = None
        for chunk_id in context_chunk_ids:
            normalized_chunk = normalize_for_evidence(chunk_by_id[chunk_id])
            if normalized_evidence and normalized_evidence in normalized_chunk:
                best_chunk, best_recall, match_method = chunk_id, 1.0, "normalized_contains"
                break
            recall = token_recall(evidence, chunk_by_id[chunk_id])
            if recall > best_recall:
                best_chunk, best_recall = chunk_id, recall
        covered = match_method is not None or best_recall >= threshold
        if covered and match_method is None:
            match_method = "token_recall"
        details.append(
            {
                "covered": covered,
                "chunk_id": best_chunk if covered else None,
                "match_method": match_method,
                "best_token_recall": round(best_recall, 8),
            }
        )
    score = sum(item["covered"] for item in details) / len(details)
    return round(float(score), 8), details


def extract_json_payload(text: str) -> str:
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    for fenced in re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.I):
        try:
            json.loads(fenced.strip())
            return fenced.strip()
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end]
    return text


@dataclass
class _Message:
    content: str


def normalize_official_classification(payload: Any) -> dict[str, list[dict[str, str]]]:
    """Normalize judge-equivalent TP/FP/FN JSON without changing class counts."""
    if isinstance(payload, dict):
        for wrapper in ("classification", "classifications", "result", "output"):
            wrapped = payload.get(wrapper)
            if isinstance(wrapped, dict):
                payload = wrapped
                break
            if isinstance(wrapped, list):
                payload = wrapped
                break
    if isinstance(payload, list):
        grouped: dict[str, list[Any]] = {"TP": [], "FP": [], "FN": []}
        for item in payload:
            if not isinstance(item, dict):
                raise Phase5Error("official judge categorized item is invalid")
            category = str(
                item.get("category") or item.get("classification") or item.get("label") or ""
            ).upper()
            aliases = {
                "TRUE_POSITIVE": "TP",
                "FALSE_POSITIVE": "FP",
                "FALSE_NEGATIVE": "FN",
            }
            category = aliases.get(category, category)
            if category not in grouped:
                raise Phase5Error("official judge categorized item has no TP/FP/FN label")
            grouped[category].append(item)
        payload = grouped
    if not isinstance(payload, dict):
        raise Phase5Error("official judge classification schema is invalid")
    key_aliases = {
        "TP": ("TP", "TRUE_POSITIVES", "TRUE_POSITIVE"),
        "FP": ("FP", "FALSE_POSITIVES", "FALSE_POSITIVE"),
        "FN": ("FN", "FALSE_NEGATIVES", "FALSE_NEGATIVE"),
    }
    uppercase = {str(key).upper(): value for key, value in payload.items()}
    normalized: dict[str, list[dict[str, str]]] = {}
    for destination, aliases in key_aliases.items():
        found = next((uppercase[key] for key in aliases if key in uppercase), None)
        if found is None:
            raise Phase5Error("official judge classification schema is invalid")
        if isinstance(found, int) and not isinstance(found, bool) and found >= 0:
            found = [
                {
                    "statement": f"Judge {destination} item {index + 1}.",
                    "reason": "Judge returned only the category count.",
                }
                for index in range(found)
            ]
        elif isinstance(found, (str, dict)):
            found = [found]
        if not isinstance(found, list):
            raise Phase5Error("official judge classification values must be arrays")
        normalized_items = []
        for item in found:
            if isinstance(item, str) and item.strip():
                normalized_items.append(
                    {"statement": item.strip(), "reason": "Reason omitted by judge."}
                )
            elif isinstance(item, dict):
                statement = item.get("statement") or item.get("claim") or item.get("text")
                if not isinstance(statement, str) or not statement.strip():
                    raise Phase5Error("official judge classification item is invalid")
                normalized_items.append(
                    {
                        "statement": statement.strip(),
                        "reason": str(item.get("reason") or "Reason omitted by judge.").strip(),
                    }
                )
            else:
                raise Phase5Error("official judge classification item is invalid")
        normalized[destination] = normalized_items
    return normalized


class CachedOfficialJudge:
    """Duck-typed LLM adapter for GraphRAG-Bench's official metric code."""

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient | None,
        cache_dir: Path,
        model: str,
        generation_config: Mapping[str, Any],
        offline: bool,
    ) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.model = model
        self.config = dict(generation_config)
        self.offline = offline
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_calls = 0
        self.cache_hits = 0
        self.successful_calls = 0
        self.unique_usage = collections.Counter()

    async def ainvoke(self, prompt: str, config: Any = None) -> _Message:
        is_classification = "Given a ground truth and an answer statements" in prompt
        kind = "classification" if is_classification else "statement_generation"
        system = (
            "Return only one valid JSON object with exactly TP, FP, and FN arrays. "
            "Do not output Markdown or text outside JSON."
            if is_classification
            else "Return only the JSON array requested by the user. Do not output Markdown."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        prompt_hash = sha256_text(canonical_json(messages))
        request_fingerprint = generation_request_fingerprint(
            prompt_hash, self.model, self.config
        )
        cache_path = self.cache_dir / f"{kind}__{prompt_hash}.json"
        if cache_path.exists():
            cache = read_json(cache_path)
            if (
                cache.get("cache_version") == EVALUATION_CACHE_VERSION
                and cache.get("prompt_sha256") == prompt_hash
                and cache.get("request_fingerprint") == request_fingerprint
                and cache.get("model") == self.model
                and isinstance(cache.get("normalized_content"), str)
            ):
                self.cache_hits += 1
                return _Message(cache["normalized_content"])
        if self.offline:
            raise Phase5Error("offline evaluation lacks a valid official-judge cache")
        if self.client is None:
            raise Phase5Error("official judge client is not initialized")
        failure_path = self.cache_dir / f"{kind}__{prompt_hash}.failure.json"
        attempts: list[dict[str, Any]] = []
        raw_responses: list[dict[str, Any]] = []
        for attempt in range(int(self.config["max_retries"]) + 1):
            self.api_calls += 1
            try:
                call_messages = list(messages)
                if is_classification and (attempt > 0 or failure_path.exists()):
                    current_analysis = prompt.split("Current Analysis:", 1)[-1]
                    call_messages = [
                        {
                            "role": "system",
                            "content": "You are a strict fact classification JSON API. "
                            'Output exactly {"TP": n, "FP": n, "FN": n} with integer counts.',
                        },
                        {
                            "role": "user",
                            "content": "Compare the answer statements with the ground-truth statements. "
                            "TP is the number of answer statements directly supported by ground truth. "
                            "FP is the number of answer statements not directly supported by ground truth. "
                            "FN is the number of ground-truth statements missing from the answer.\n\n"
                            + current_analysis
                            + "\n\nReturn only one JSON object, never an array or prose. "
                            + 'Required format: {"TP": 0, "FP": 0, "FN": 0}.',
                        },
                    ]
                response, elapsed = self.client.call_once(
                    call_messages, strict_object=is_classification
                )
                raw_responses.append(response)
                normalized = extract_json_payload(response_content(response))
                payload = json.loads(normalized)
                if is_classification:
                    normalized = canonical_json(normalize_official_classification(payload))
                elif not isinstance(payload, list) or not all(
                    isinstance(item, str) for item in payload
                ):
                    raise Phase5Error("official judge statement generation must be a string array")
                usage = usage_from_response(response)
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "success",
                        "elapsed_ms": round(elapsed * 1000, 6),
                        "token_usage": usage,
                    }
                )
                cache = {
                    "cache_version": EVALUATION_CACHE_VERSION,
                    "kind": kind,
                    "prompt_sha256": prompt_hash,
                    "request_fingerprint": request_fingerprint,
                    "model": self.model,
                    "normalized_content": normalized,
                    "request_audit": {"attempts": attempts, "token_usage": usage},
                    "raw_response": response,
                }
                write_text_atomic(cache_path, canonical_json(cache, pretty=True))
                self.successful_calls += 1
                self.unique_usage.update(usage)
                return _Message(normalized)
            except Exception as exc:
                if isinstance(exc, PermanentLLMError):
                    raise
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:800],
                    }
                )
                if attempt < int(self.config["max_retries"]):
                    time.sleep(float(self.config["retry_backoff_seconds"]) * (attempt + 1))
        write_text_atomic(
            failure_path,
            canonical_json(
                {
                    "cache_version": EVALUATION_CACHE_VERSION,
                    "kind": kind,
                    "prompt_sha256": prompt_hash,
                    "request_fingerprint": request_fingerprint,
                    "model": self.model,
                    "outcome": "failure",
                    "attempts": attempts,
                    "raw_responses": raw_responses,
                },
                pretty=True,
            ),
        )
        raise Phase5Error(f"official judge failed after retries: {attempts[-1]['error']}")


class OfficialBGEEmbeddings:
    def __init__(self, encoder: BGEEncoder) -> None:
        self.encoder = encoder
        self.cache: dict[str, list[float]] = {}

    async def aembed_query(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.encoder.encode_documents([text])[0].tolist()
        return self.cache[text]


def load_official_answer_correctness(benchmark_dir: Path) -> tuple[Any, str]:
    metric_path = benchmark_dir / "Evaluation" / "metrics" / "answer_accuracy.py"
    if not metric_path.is_file():
        raise Phase5Error(
            "GraphRAG-Bench official answer_accuracy.py is absent; no silent metric substitution"
        )
    benchmark_text = str(benchmark_dir)
    if benchmark_text not in sys.path:
        sys.path.insert(0, benchmark_text)
    try:
        from Evaluation.metrics.answer_accuracy import compute_answer_correctness
    except ImportError as exc:
        raise Phase5Error(
            "GraphRAG-Bench official metric dependencies are unavailable; use the existing official environment"
        ) from exc
    return compute_answer_correctness, sha256_file(metric_path)


async def official_correctness_for_references(
    *,
    metric: Any,
    question: str,
    answer: str,
    references: Sequence[str],
    judge: CachedOfficialJudge,
    embeddings: OfficialBGEEmbeddings,
    weights: Sequence[float],
    beta: float,
) -> float:
    scores = []
    for reference in references:
        score = await metric(
            question,
            answer,
            reference,
            judge,
            embeddings,
            weights=list(weights),
            beta=beta,
        )
        scores.append(min(1.0, max(0.0, float(score))))
    return round(max(scores), 8)


def validate_train_answers(
    rows: Sequence[Mapping[str, Any]], retrieval_rows: Sequence[Mapping[str, Any]]
) -> None:
    if len(rows) != 186:
        raise Phase5Error("train_answers.jsonl must contain 186 rows")
    answer_keys = [(row.get("query_id"), row.get("retriever")) for row in rows]
    retrieval_keys = [(row.get("query_id"), row.get("retriever")) for row in retrieval_rows]
    if answer_keys != retrieval_keys or len(set(answer_keys)) != 186:
        raise Phase5Error("train answers do not align exactly with frozen retrieval results")
    for row in rows:
        if SUPERVISION_FIELDS & set(row):
            raise Phase5Error("supervision leaked into train answer rows")
        if len(row.get("context_chunk_ids", [])) > 5:
            raise Phase5Error("generation row exceeds the five-chunk context budget")
        invalid = set(row.get("citations", [])) - set(row.get("context_chunk_ids", []))
        if sorted(invalid) != sorted(row.get("invalid_citations", [])):
            raise Phase5Error("invalid citation audit is inconsistent")


def choose_router_label(
    method_rows: Mapping[str, Mapping[str, Any]], difficulty: Mapping[str, int]
) -> tuple[str, str, list[str]]:
    acceptable = [
        method for method in RETRIEVERS if bool(method_rows[method]["acceptable"])
    ]
    if acceptable:
        selected = min(acceptable, key=lambda method: difficulty[method])
        return selected, "acceptable", acceptable
    selected = min(
        RETRIEVERS,
        key=lambda method: (
            -float(method_rows[method]["answer_correctness"]),
            -float(method_rows[method]["evidence_coverage"]),
            int(method_rows[method]["total_tokens"]),
            int(difficulty[method]),
        ),
    )
    return selected, "best_available", []


def summarize_methods(
    scores: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    win_counts = collections.Counter(row["label"] for row in labels)
    methods: dict[str, Any] = {}
    for method in RETRIEVERS:
        rows = [row for row in scores if row["retriever"] == method]
        methods[method] = {
            "count": len(rows),
            "mean_answer_correctness": round(
                statistics.fmean(float(row["answer_correctness"]) for row in rows), 8
            ),
            "mean_evidence_coverage": round(
                statistics.fmean(float(row["evidence_coverage"]) for row in rows), 8
            ),
            "acceptable_count": sum(bool(row["acceptable"]) for row in rows),
            "acceptable_rate": round(
                sum(bool(row["acceptable"]) for row in rows) / len(rows), 8
            ),
            "win_count": int(win_counts[method]),
            "mean_prompt_tokens": round(
                statistics.fmean(int(row["prompt_tokens"]) for row in rows), 4
            ),
            "mean_completion_tokens": round(
                statistics.fmean(int(row["completion_tokens"]) for row in rows), 4
            ),
            "mean_total_tokens": round(
                statistics.fmean(int(row["total_tokens"]) for row in rows), 4
            ),
            "total_tokens": sum(int(row["total_tokens"]) for row in rows),
            "mean_context_character_count": round(
                statistics.fmean(int(row["context_character_count"]) for row in rows), 4
            ),
            "mean_retrieval_latency_ms": round(
                statistics.fmean(float(row["retrieval_latency_ms"]) for row in rows), 6
            ),
            "mean_generation_latency_ms": round(
                statistics.fmean(float(row["generation_latency_ms"]) for row in rows), 6
            ),
            "insufficient_or_failed_count": sum(
                row["evaluation_status"] != "scored" for row in rows
            ),
            "invalid_citation_count": sum(int(row["invalid_citation_count"]) for row in rows),
        }
    label_counts = {method: int(win_counts[method]) for method in RETRIEVERS}
    return {
        "schema_version": "phase5-v1",
        "question_count": len(labels),
        "method_score_count": len(scores),
        "methods": methods,
        "label_distribution": label_counts,
        "selection_modes": dict(
            sorted(collections.Counter(row["selection_mode"] for row in labels).items())
        ),
    }


def render_report(
    summary: Mapping[str, Any], manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> str:
    rows = []
    for method in RETRIEVERS:
        item = summary["methods"][method]
        rows.append(
            f"| {method} | {item['mean_answer_correctness']:.4f} | "
            f"{item['mean_evidence_coverage']:.4f} | {item['acceptable_rate']:.2%} | "
            f"{item['mean_context_character_count']:.1f} | "
            f"{item['mean_prompt_tokens']:.1f} | {item['mean_completion_tokens']:.1f} | "
            f"{item['mean_total_tokens']:.1f} | {item['mean_retrieval_latency_ms']:.3f} | "
            f"{item['mean_generation_latency_ms']:.1f} | "
            f"{item['invalid_citation_count']} | {item['win_count']} |"
        )
    risks = manifest["class_imbalance_warnings"]
    risk_text = "无。" if not risks else "；".join(risks)
    return f"""# 阶段 5：训练集答案生成、评估与伪标签

## 协议边界

本阶段只使用冻结的 186 条 phase4 训练检索结果、62 个 phase2 Chunk 和 62 条
phase1 训练问题。生成阶段只投影问题 ID 与问题正文，生成全部结束后评估脚本才
加载标准答案和标准证据。未读取或运行测试集，未重跑检索器，未训练分类器。

三种方法统一使用 `{manifest['generation']['model']}`、temperature=0、thinking
关闭、同一系统 Prompt、最多 5 个 Chunk。Path 的 10 条非 success 结果未调用
LLM，直接记为证据不足。Neighborhood/Path 只附加 phase4 结果中已有的图上下文。

Answer Correctness 直接复用固定 GraphRAG-Bench 官方实现：LLM 将答案与参考答案
拆为陈述并计算事实性 F1（权重 0.75），BGE 余弦相似度映射至 [0,1]（权重
0.25）；多参考答案取最高分。证据覆盖按归一化包含匹配，失败时以 token recall
≥{config['evaluation']['evidence_token_recall_threshold']} 判为覆盖。

## 方法结果

| 方法 | 平均正确性 | 平均证据覆盖 | 可接受率 | 上下文字符 | Prompt tok | Completion tok | 总 tok | 检索 ms | 生成 ms | 无效引用 | 胜出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

阈值在运行前固定为 Answer Correctness ≥
{config['labeling']['answer_correctness_threshold']} 且 Evidence Coverage ≥
{config['labeling']['evidence_coverage_threshold']}。可接受时按 vector、neighborhood、
path 的难度顺序选最简单方法；三者均不可接受时依次比较正确性、证据覆盖、总
tokens 和难度，并保留为 `best_available`。

## 标签与审计

- 标签分布：`{summary['label_distribution']}`
- 选择模式：`{summary['selection_modes']}`
- 类别失衡风险：{risk_text}
- 生成有效缓存：{manifest['generation']['successful_cache_count']}；直接跳过：{manifest['generation']['retrieval_failure_skip_count']}
- 最后一次离线生成重建 API 调用：{manifest['generation']['offline_replay_api_calls']}
- 官方评审有效缓存：{manifest['evaluation']['judge_cache_count']}；已解决的结构失败审计缓存：{manifest['evaluation']['judge_failure_audit_count']}；最后一次离线评估 API 调用：{manifest['evaluation']['offline_replay_api_calls']}
- 官方评审有效缓存 token：`{manifest['evaluation']['judge_cached_token_usage']}`
- 输入 SHA-256：`{manifest['inputs']}`

非耗时产物已通过重复运行一致性验证。阶段 1～4 产物未修改。下一步可训练路由
分类器，但本阶段按要求在此停止。
"""


def evaluate_and_build_labels(
    root: Path,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    offline: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    input_hashes = validate_frozen_inputs(config, paths)
    if not paths["train_answers"].is_file():
        raise Phase5Error("train answers must be generated before supervision is loaded")

    questions = load_generation_questions(paths["train_questions"])
    _chunks, chunk_by_id = load_chunks(
        paths["chunks"], config["inputs"]["chunks"]["expected_count"]
    )
    retrieval_rows = load_and_validate_retrieval_results(
        paths["retrieval_results"],
        questions,
        chunk_by_id,
        config["inputs"]["retrieval_results"]["expected_count"],
        config["max_context_chunks"],
    )
    answer_rows = read_jsonl(paths["train_answers"])
    validate_train_answers(answer_rows, retrieval_rows)

    # Supervision is intentionally first loaded only after all generation rows
    # have passed the completeness and prompt-isolation checks above.
    references = load_evaluation_references(paths["train_questions"])
    metric, metric_sha = load_official_answer_correctness(paths["benchmark_directory"])
    generation_cache_dir = paths["generation_responses"]
    runtime = runtime_from_env(online=not offline, cache_dir=generation_cache_dir)
    client = None if offline else OpenAICompatibleClient(runtime, config["generation"])
    judge = CachedOfficialJudge(
        client=client,
        cache_dir=paths["evaluation_responses"],
        model=runtime["LLM_MODEL"],
        generation_config=config["generation"],
        offline=offline,
    )
    encoder = BGEEncoder(config["evaluation"]["embedding"], config["random_seed"])
    embeddings = OfficialBGEEmbeddings(encoder)
    weights = [
        float(config["evaluation"]["factuality_weight"]),
        float(config["evaluation"]["semantic_weight"]),
    ]
    beta = float(config["evaluation"]["factuality_beta"])
    evidence_threshold = float(config["evaluation"]["evidence_token_recall_threshold"])

    scores: list[dict[str, Any]] = []
    for position, answer_row in enumerate(answer_rows, 1):
        reference = references[answer_row["query_id"]]
        coverage, coverage_details = evidence_coverage(
            reference["evidence"],
            answer_row["context_chunk_ids"],
            chunk_by_id,
            evidence_threshold,
        )
        if answer_row["generation_status"] != "success":
            correctness = 0.0
            evaluation_status = (
                "zero_retrieval_failure"
                if answer_row["generation_status"] == "retrieval_failure"
                else "zero_insufficient_or_generation_failure"
            )
        else:
            correctness = asyncio.run(
                official_correctness_for_references(
                    metric=metric,
                    question=answer_row["query"],
                    answer=answer_row["answer"],
                    references=reference["answers"],
                    judge=judge,
                    embeddings=embeddings,
                    weights=weights,
                    beta=beta,
                )
            )
            evaluation_status = "scored"
        acceptable = (
            correctness >= float(config["labeling"]["answer_correctness_threshold"])
            and coverage >= float(config["labeling"]["evidence_coverage_threshold"])
        )
        generation = answer_row["generation"]
        scores.append(
            {
                "schema_version": "phase5-v1",
                "query_id": answer_row["query_id"],
                "retriever": answer_row["retriever"],
                "retrieval_status": answer_row["retrieval_status"],
                "generation_status": answer_row["generation_status"],
                "evaluation_status": evaluation_status,
                "answer_correctness": correctness,
                "evidence_coverage": coverage,
                "evidence_item_count": len(reference["evidence"]),
                "covered_evidence_count": sum(item["covered"] for item in coverage_details),
                "acceptable": acceptable,
                "retrieval_latency_ms": answer_row["retrieval_latency_ms"],
                "generation_latency_ms": generation["elapsed_ms"],
                "prompt_tokens": generation["prompt_tokens"],
                "completion_tokens": generation["completion_tokens"],
                "total_tokens": generation["total_tokens"],
                "context_character_count": answer_row["context_character_count"],
                "citation_count": len(answer_row["citations"]),
                "invalid_citation_count": len(answer_row["invalid_citations"]),
                "metric": "GraphRAG-Bench official Answer Correctness",
            }
        )
        print(
            f"[{position:03d}/186] {answer_row['query_id']}/{answer_row['retriever']}: "
            f"correctness={correctness:.4f} coverage={coverage:.4f}",
            flush=True,
        )

    score_by_question: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in scores:
        score_by_question[row["query_id"]][row["retriever"]] = row
    question_text = {row["id"]: row["question"] for row in questions}
    labels: list[dict[str, Any]] = []
    difficulty = config["labeling"]["difficulty_order"]
    for query_id in [row["id"] for row in questions]:
        methods = score_by_question[query_id]
        if set(methods) != set(RETRIEVERS):
            raise Phase5Error(f"incomplete method scores for {query_id}")
        selected, selection_mode, acceptable_methods = choose_router_label(methods, difficulty)
        labels.append(
            {
                "schema_version": "phase5-v1",
                "query_id": query_id,
                "query": question_text[query_id],
                "label": selected,
                "difficulty": difficulty[selected],
                "selection_mode": selection_mode,
                "selection_reason": (
                    "simplest acceptable method"
                    if selection_mode == "acceptable"
                    else "best available by correctness, coverage, total tokens, then difficulty"
                ),
                "acceptable_methods": acceptable_methods,
                "method_scores": {
                    method: {
                        "answer_correctness": methods[method]["answer_correctness"],
                        "evidence_coverage": methods[method]["evidence_coverage"],
                        "total_tokens": methods[method]["total_tokens"],
                        "acceptable": methods[method]["acceptable"],
                        "evaluation_status": methods[method]["evaluation_status"],
                    }
                    for method in RETRIEVERS
                },
            }
        )
    if len(labels) != 62 or len({row["query_id"] for row in labels}) != 62:
        raise Phase5Error("router labels must contain exactly 62 unique questions")
    summary = summarize_methods(scores, labels)
    warning_count = int(config["labeling"]["minimum_class_count_warning"])
    warning_rate = float(config["labeling"]["minimum_class_rate_warning"])
    imbalance = []
    for method, count in summary["label_distribution"].items():
        rate = count / len(labels)
        if count < warning_count or rate < warning_rate:
            imbalance.append(f"{method}: {count}/62 ({rate:.2%})")

    write_text_atomic(paths["method_scores"], jsonl_dump(scores))
    write_text_atomic(paths["router_labels"], jsonl_dump(labels))
    write_text_atomic(paths["method_summary"], canonical_json(summary, pretty=True))
    evaluation_audit = {
        "schema_version": "phase5-v1",
        "mode": "offline" if offline else "online",
        "model": runtime["LLM_MODEL"],
        "api_calls_this_run": judge.api_calls,
        "successful_api_calls_this_run": judge.successful_calls,
        "cache_hits_this_run": judge.cache_hits,
        "unique_api_usage_this_run": dict(judge.unique_usage),
        "method_scores_sha256": sha256_file(paths["method_scores"]),
        "router_labels_sha256": sha256_file(paths["router_labels"]),
        "method_summary_sha256": sha256_file(paths["method_summary"]),
    }
    write_text_atomic(paths["evaluation_run_audit"], canonical_json(evaluation_audit, pretty=True))

    generation_audit = read_json(paths["generation_run_audit"])
    generation_caches = list(paths["generation_responses"].glob("*.json"))
    all_evaluation_caches = list(paths["evaluation_responses"].glob("*.json"))
    valid_evaluation_caches = []
    failed_evaluation_caches = []
    evaluation_cache_kinds: collections.Counter[str] = collections.Counter()
    evaluation_cached_usage: collections.Counter[str] = collections.Counter()
    classification_shapes: collections.Counter[str] = collections.Counter()
    for cache_path in all_evaluation_caches:
        cached = read_json(cache_path)
        if cached.get("outcome") == "failure" or cache_path.name.endswith(".failure.json"):
            failed_evaluation_caches.append(cache_path)
            continue
        if not isinstance(cached.get("normalized_content"), str):
            continue
        valid_evaluation_caches.append(cache_path)
        evaluation_cache_kinds[str(cached.get("kind"))] += 1
        evaluation_cached_usage.update(
            {
                key: int(value)
                for key, value in cached.get("request_audit", {})
                .get("token_usage", {})
                .items()
            }
        )
        if cached.get("kind") == "classification":
            try:
                raw_content = response_content(cached["raw_response"])
                raw_payload = json.loads(extract_json_payload(raw_content))
                if isinstance(raw_payload, dict) and all(
                    isinstance(raw_payload.get(key), int) for key in ("TP", "FP", "FN")
                ):
                    classification_shapes["count_object"] += 1
                elif isinstance(raw_payload, dict) and all(
                    isinstance(raw_payload.get(key), list) for key in ("TP", "FP", "FN")
                ):
                    classification_shapes["official_array_object"] += 1
                else:
                    classification_shapes["compatible_other"] += 1
            except (KeyError, TypeError, json.JSONDecodeError, Phase5Error):
                classification_shapes["unknown"] += 1
    manifest = {
        "schema_version": "phase5-v1",
        "scope": "training-only generation, evaluation, and pseudo-labeling",
        "inputs": {
            key: {"path": config["inputs"][key]["path"], "sha256": value}
            for key, value in input_hashes.items()
        },
        "config_sha256": sha256_file(paths["config"]),
        "system_prompt_sha256": sha256_file(paths["system_prompt"]),
        "counts": {"questions": 62, "train_answers": 186, "method_scores": 186, "router_labels": 62},
        "generation": {
            "model": runtime["LLM_MODEL"],
            "temperature": config["generation"]["temperature"],
            "thinking": config["generation"]["thinking"],
            "max_retries": config["generation"]["max_retries"],
            "max_context_chunks": config["max_context_chunks"],
            "successful_cache_count": sum(
                read_json(path).get("outcome") == "success" for path in generation_caches
            ),
            "retrieval_failure_skip_count": sum(
                row["generation_status"] == "retrieval_failure" for row in answer_rows
            ),
            "offline_replay_api_calls": generation_audit["api_calls_this_run"]
            if generation_audit.get("mode") == "offline"
            else None,
        },
        "evaluation": {
            "answer_correctness_implementation": "GraphRAG-Bench official",
            "official_metric_sha256": metric_sha,
            "factuality_weight": weights[0],
            "semantic_weight": weights[1],
            "embedding": config["evaluation"]["embedding"],
            "evidence_token_recall_threshold": evidence_threshold,
            "judge_cache_count": len(valid_evaluation_caches),
            "judge_failure_audit_count": len(failed_evaluation_caches),
            "judge_cache_types": dict(sorted(evaluation_cache_kinds.items())),
            "judge_cached_token_usage": dict(sorted(evaluation_cached_usage.items())),
            "classification_response_shapes": dict(sorted(classification_shapes.items())),
            "offline_replay_api_calls": evaluation_audit["api_calls_this_run"]
            if offline
            else None,
        },
        "thresholds": config["labeling"],
        "label_distribution": summary["label_distribution"],
        "class_imbalance_warnings": imbalance,
        "outputs": {
            key: {
                "path": paths[key].relative_to(root).as_posix(),
                "sha256": sha256_file(paths[key]),
            }
            for key in ("train_answers", "method_scores", "router_labels", "method_summary")
        },
        "test_set_accessed": False,
        "retrievers_rerun": False,
        "router_trained": False,
        "standard_answers_in_generation_prompt": False,
    }
    write_text_atomic(paths["manifest"], canonical_json(manifest, pretty=True))
    write_text_atomic(paths["report"], render_report(summary, manifest, config))
    if validate_frozen_inputs(config, paths) != input_hashes:
        raise Phase5Error("a frozen phase-one, phase-two, or phase-four input changed")
    return scores, labels, summary, manifest
