import argparse
import asyncio
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.adapters.lightrag import LightRAGAdapter
from app.config import DEFAULT_BENCHMARK_ROOT, Settings
from app.domain.errors import AppError
from app.domain.models import QueryMode, QueryResult
from app.services.dataset_service import DatasetService
from app.adapters.graphrag_bench import GraphRAGBenchLoader
from app.services.lightrag_factory import LightRAGFactory
from app.services.workspace_service import WorkspaceService


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = (
    DEFAULT_BENCHMARK_ROOT / "Datasets" / "Questions" / "novel_questions.json"
)
QUESTION_TYPE_ORDER = (
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
)
OFFICIAL_SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Knowledge Base---
{context_data}
""".strip()


def select_balanced_questions(
    questions: Sequence[dict[str, Any]],
    sample_size: int,
) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise AppError("INVALID_SAMPLE", "Sample size must be at least 1")
    queues = {
        question_type: deque(
            question
            for question in questions
            if question.get("question_type") == question_type
        )
        for question_type in QUESTION_TYPE_ORDER
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < sample_size:
        added = False
        for question_type in QUESTION_TYPE_ORDER:
            if queues[question_type] and len(selected) < sample_size:
                selected.append(queues[question_type].popleft())
                added = True
        if not added:
            break
    return selected


def build_prediction(
    question: dict[str, Any],
    result: QueryResult,
) -> dict[str, Any]:
    contexts = [
        chunk["content"]
        for chunk in result.chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("content"), str)
        and chunk["content"]
    ]
    return {
        "id": question["id"],
        "question": question["question"],
        "source": question["source"],
        "context": contexts,
        "evidence": question["evidence"],
        "question_type": question["question_type"],
        "generated_answer": result.answer or "",
        "ground_truth": question.get("answer"),
    }


def load_question_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AppError("INVALID_QUESTIONS", "The questions file cannot be read") from exc
    elif path.suffix.lower() == ".parquet":
        try:
            from datasets import load_dataset

            rows = list(
                load_dataset("parquet", data_files=str(path), split="train")
            )
        except Exception as exc:
            raise AppError("INVALID_QUESTIONS", "The questions file cannot be read") from exc
    else:
        raise AppError(
            "UNSUPPORTED_QUESTIONS",
            "Only JSON and Parquet question files are supported",
        )
    if not isinstance(rows, list):
        raise AppError("INVALID_QUESTIONS", "The questions file must contain a list")
    return rows


def write_predictions(path: Path, predictions: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(list(predictions), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


async def run(args: argparse.Namespace) -> Path:
    settings = Settings(app_root=APP_ROOT)
    dataset_service = DatasetService(GraphRAGBenchLoader(), settings.active_dir)
    document = dataset_service.load_active()
    if document is None:
        raise AppError("NO_ACTIVE_DATASET", "No active dataset is available")

    rows = load_question_rows(args.questions)
    source_rows = [row for row in rows if row.get("source") == document.corpus_name]
    selected = select_balanced_questions(source_rows, args.sample)
    if not selected:
        raise AppError("NO_BENCHMARK_QUESTIONS", "No questions match the active corpus")

    factory = LightRAGFactory(settings)
    workspace_service = WorkspaceService(
        settings.workspace_dir,
        factory,
        factory.runtime_fingerprint(),
    )
    outcome = await workspace_service.ensure_ready(document)
    if outcome.action != "reused":
        raise AppError("INDEX_NOT_REUSED", "Benchmark queries require a ready workspace")

    output = args.output or (
        APP_ROOT
        / "artifacts"
        / "benchmark"
        / f"{document.corpus_name}-{args.mode}-{len(selected)}.json"
    )
    predictions: list[dict[str, Any]] = []
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            selected_ids = {question["id"] for question in selected}
            predictions = [
                item
                for item in existing
                if isinstance(item, dict) and item.get("id") in selected_ids
            ]
    completed_ids = {prediction["id"] for prediction in predictions}

    rag = await factory.create()
    try:
        for position, question in enumerate(selected, start=1):
            if question["id"] in completed_ids:
                print(f"[{position}/{len(selected)}] reused prediction {question['id']}")
                continue
            result = await LightRAGAdapter().query(
                rag,
                question["question"],
                QueryMode(args.mode),
                args.top_k,
                system_prompt=OFFICIAL_SYSTEM_PROMPT,
            )
            predictions.append(build_prediction(question, result))
            write_predictions(output, predictions)
            print(f"[{position}/{len(selected)}] saved prediction {question['id']}")
    finally:
        await rag.finalize_storages()
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a balanced GraphRAG-Bench prediction subset."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in QueryMode],
        default=QueryMode.MIX.value,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output = asyncio.run(run(parse_args(argv)))
    except AppError as exc:
        print(
            json.dumps(
                {"status": "failure", "error": {"code": exc.code, "message": exc.safe_message}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failure",
                    "error": {"code": "UNEXPECTED_ERROR", "message": type(exc).__name__},
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"status": "success", "predictions": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
