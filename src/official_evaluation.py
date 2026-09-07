from __future__ import annotations

import asyncio
import json
import hashlib
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVALUATION_VERSION = 2
METRICS = ("rouge_l", "answer_correctness", "evidence_recall")


class JudgeResponseError(ValueError):
    pass


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def valid_evaluation(row: dict[str, Any]) -> bool:
    return (row.get("evaluation", {}).get("version") == EVALUATION_VERSION
            and row.get("evaluation", {}).get("status") == "ok"
            and all(isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool)
                    and math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in METRICS))


def parse_judge_json(text: str, kind: str, evidence: list[str]) -> Any:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].lower() not in {"```", "```json"} or lines[-1] != "```":
            raise JudgeResponseError("Incomplete JSON fence")
        text = "\n".join(lines[1:-1])
    data = json.loads(text)
    if kind == "statements":
        if not isinstance(data, list) or not data or not all(
            isinstance(s, str) and s.strip() for s in data
        ):
            raise JudgeResponseError("Expected a nonempty list of statements")
        return data
    keys = ("TP", "FP", "FN") if kind == "correctness" else ("classifications",)
    if not isinstance(data, dict) or any(k not in data for k in keys):
        raise JudgeResponseError(f"Missing required fields: {keys}")
    for key in keys:
        if not isinstance(data[key], list):
            raise JudgeResponseError(f"{key} must be a list")
        for item in data[key]:
            if not isinstance(item, dict) or not all(
                isinstance(item.get(k), str) and item[k].strip()
                for k in ("statement", "reason")
            ):
                raise JudgeResponseError("Invalid statement/reason")
            if kind == "evidence" and (type(item.get("attributed")) is not int
                                       or item["attributed"] not in (0, 1)):
                raise JudgeResponseError("attributed must be 0 or 1")
    if kind == "correctness" and not any(data[k] for k in keys):
        raise JudgeResponseError("Classification cannot be entirely empty")
    if kind == "evidence":
        # Judge 可能对 reference evidence 做无害标点改写(如 t(11;22)→t(11, 22))。
        # 逐字 Counter 比对会把这类改写误判为不匹配;归一化(去空白/分号→逗号/小写)后再比对。
        def _norm_evidence(s: str) -> str:
            return re.sub(r"\s+", "", str(s)).replace(";", ",").lower()

        if Counter(
            _norm_evidence(x["statement"]) for x in data["classifications"]
        ) != Counter(_norm_evidence(e) for e in evidence):
            raise JudgeResponseError(
                "Classifications must cover every reference evidence exactly once"
            )
    return data


@dataclass
class _Message:
    content: str


class LocalJudgeLLM:
    """Small duck-typed adapter for GraphRAG-Bench's official metric code."""

    def __init__(self, client: Any, evidence: list[str] | None = None,
                 max_tokens: int = 4096, retries: int = 1) -> None:
        self.client = client
        self.evidence = evidence or []
        self.max_tokens = max_tokens
        self.retries = retries
        self.trace: list[dict[str, Any]] = []
        self.failed_prompts: dict[str, str] = {}

    async def ainvoke(self, prompt: str, config: Any = None) -> _Message:
        if prompt in self.failed_prompts:
            raise JudgeResponseError(self.failed_prompts[prompt])
        if prompt.lstrip().startswith("Given a question and an answer,"):
            kind = "statements"
        elif prompt.lstrip().startswith("Given a ground truth and an answer statements,"):
            kind = "correctness"
        elif prompt.lstrip().startswith("### Task"):
            kind = "evidence"
        else:
            raise JudgeResponseError("Unrecognized official metric prompt")
        for attempt in range(self.retries + 1):
            entry = {"prompt": prompt, "kind": kind, "attempt": attempt + 1}
            self.trace.append(entry)
            print(f"    judge {kind}: attempt {attempt + 1}", flush=True)
            # Keep the upstream JSON array schema for statement extraction.
            response = await self.client.generate(
                prompt, max_new_tokens=self.max_tokens,
                system_prompt="Return only complete valid JSON matching the requested schema. No Markdown fences.",
            )
            entry.update(response=response.text, finish_reason=response.finish_reason,
                         output_tokens=response.output_tokens)
            try:
                if response.finish_reason not in (None, "stop") or response.output_tokens >= self.max_tokens:
                    raise JudgeResponseError("Incomplete or non-normal completion")
                data = parse_judge_json(response.text, kind, self.evidence)
                entry["status"] = "ok"
                return _Message(content=json.dumps(data, ensure_ascii=False, allow_nan=False))
            except (ValueError, TypeError) as exc:
                entry.update(status="invalid", error=str(exc))
        self.failed_prompts[prompt] = f"Invalid {kind} response after {self.retries + 1} attempts"
        raise JudgeResponseError(self.failed_prompts[prompt])


class LocalJudgeEmbeddings:
    def __init__(self, client: Any) -> None:
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
    judge_client: Any,
    embedding_client: Any,
    benchmark_dir: Path,
    *,
    max_tokens: int = 4096,
    retries: int = 1,
    protocol: str = "",
) -> dict[str, Any]:
    answer_correctness, evidence_recall, rouge_score = _load_official_metrics(
        benchmark_dir
    )
    evidence = [part.strip() for part in row.get("evidence", "").split(";") if part.strip()]
    judge = LocalJudgeLLM(judge_client, evidence, max_tokens, retries)
    embeddings = LocalJudgeEmbeddings(embedding_client)
    before = judge_client.snapshot()
    scores = await asyncio.gather(
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
            evidence,
            judge,
        ),
        return_exceptions=True,
    )
    errors = [f"{name}: {value}" for name, value in zip(METRICS, scores)
              if isinstance(value, Exception)]
    errors.extend(judge.failed_prompts.values())
    for name, value in zip(METRICS, scores):
        if not isinstance(value, Exception) and (not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
            errors.append(f"{name}: non-finite or out-of-range score")
    if not evidence:
        errors.append("Missing reference evidence")
    usage = judge_client.snapshot() - before
    return {
        **row,
        **{name: None if errors else float(value) for name, value in zip(METRICS, scores)},
        "evaluation": {"version": EVALUATION_VERSION, "status": "failed" if errors else "ok",
                       "errors": errors, "protocol": protocol, "source": fingerprint(row)},
        "judge": {
            "model": judge_client.model_name,
            "temperature": judge_client.temperature,
            "max_new_tokens": max_tokens,
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

