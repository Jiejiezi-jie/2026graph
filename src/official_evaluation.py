from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .official_backends.model_client import (
    TransformersChatClient,
    TransformersEmbeddingClient,
)


@dataclass
class _Message:
    content: str


class LocalJudgeLLM:
    """Small duck-typed adapter for GraphRAG-Bench's official metric code."""

    def __init__(self, client: TransformersChatClient) -> None:
        self.client = client
        self.trace: list[dict[str, str]] = []

    async def ainvoke(self, prompt: str, config: Any = None) -> _Message:
        response = await self.client.generate(prompt)
        self.trace.append({"prompt": prompt, "response": response.text})
        return _Message(content=response.text)


class LocalJudgeEmbeddings:
    def __init__(self, client: TransformersEmbeddingClient) -> None:
        self.client = client

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.client.embed([text]))[0].tolist()


def _load_official_metrics(benchmark_dir: Path) -> tuple[Any, Any, Any]:
    benchmark_text = str(benchmark_dir.resolve())
    if benchmark_text not in sys.path:
        sys.path.insert(0, benchmark_text)
    from Evaluation.metrics.answer_accuracy import compute_answer_correctness
    from Evaluation.metrics.evidence_recall import compute_evidence_recall
    from Evaluation.metrics.rouge import compute_rouge_score

    return compute_answer_correctness, compute_evidence_recall, compute_rouge_score


async def evaluate_official_row(
    row: dict[str, Any],
    judge_client: TransformersChatClient,
    embedding_client: TransformersEmbeddingClient,
    benchmark_dir: Path,
) -> dict[str, Any]:
    answer_correctness, evidence_recall, rouge_score = _load_official_metrics(
        benchmark_dir
    )
    judge = LocalJudgeLLM(judge_client)
    embeddings = LocalJudgeEmbeddings(embedding_client)
    before = judge_client.snapshot()
    rouge_l, correctness, recall = await asyncio.gather(
        rouge_score(row["answer"], row["ground_truth"], rouge_type="rougeL"),
        answer_correctness(
            row["question"],
            row["answer"],
            row["ground_truth"],
            judge,
            embeddings,
        ),
        evidence_recall(
            row["question"],
            row.get("contexts", []),
            [part.strip() for part in row.get("evidence", "").split(";") if part.strip()],
            judge,
        ),
    )
    usage = judge_client.snapshot() - before
    return {
        **row,
        "rouge_l": float(rouge_l),
        "answer_correctness": float(correctness),
        "evidence_recall": float(recall),
        "judge": {
            "model": judge_client.model_name,
            "temperature": judge_client.temperature,
            "same_model_family_as_generator": True,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "llm_calls": usage.calls,
            "trace": judge.trace,
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

