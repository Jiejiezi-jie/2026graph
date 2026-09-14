from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


WORD_RE = re.compile(r"\b\w+(?:[-']\w+)*\b")


def load_medical_benchmark(benchmark_dir: Path) -> tuple[str, list[dict]]:
    corpus_path = benchmark_dir / "Datasets" / "Corpus" / "medical.json"
    questions_path = benchmark_dir / "Datasets" / "Questions" / "medical_questions.json"
    if not corpus_path.exists() or not questions_path.exists():
        raise FileNotFoundError(
            "GraphRAG-Bench files are missing. Run src/backend/common/fetch_benchmark.sh or "
            "pass --benchmark-dir to the official checkout."
        )
    corpus_obj = json.loads(corpus_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    return corpus_obj["context"], questions


def select_questions(
    questions: Iterable[dict],
    question_types: set[str],
    max_per_type: int | None,
) -> list[dict]:
    selected: list[dict] = []
    counts = {name: 0 for name in question_types}
    for item in questions:
        kind = item.get("question_type")
        if kind not in question_types:
            continue
        if max_per_type is not None and counts[kind] >= max_per_type:
            continue
        selected.append(item)
        counts[kind] += 1
    return selected


def chunk_by_word_window(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    words = list(WORD_RE.finditer(text))
    if not words:
        return []
    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(words), step):
        end = min(start + size, len(words))
        chunks.append(text[words[start].start() : words[end - 1].end()])
        if end == len(words):
            break
    return chunks

