import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from app.adapters.graphrag_bench import GraphRAGBenchLoader
from app.adapters.lightrag import LightRAGAdapter
from app.config import DEFAULT_BENCHMARK_ROOT, Settings
from app.domain.errors import AppError
from app.domain.models import QueryMode, QueryResult
from app.services.dataset_service import DatasetService
from app.services.lightrag_factory import LightRAGFactory
from app.services.workspace_service import WorkspaceService


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = DEFAULT_BENCHMARK_ROOT / "Datasets" / "Corpus" / "novel.json"


def build_summary(
    corpus_name: str,
    index_action: str,
    result: QueryResult,
) -> dict[str, Any]:
    return {
        "status": "success",
        "corpus": corpus_name,
        "index_action": index_action,
        "query": result.query,
        "mode": result.mode.value,
        "answer": result.answer,
        "latency_ms": result.latency_ms,
        "context_text": result.context_text,
        "entities": result.entities,
        "relationships": result.relationships,
        "chunks": result.chunks,
        "references": result.references,
        "counts": {
            "entities": len(result.entities),
            "relationships": len(result.relationships),
            "chunks": len(result.chunks),
            "references": len(result.references),
        },
        "warnings": result.warnings,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or reuse one LightRAG workspace and run one query."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--subset", default="novel")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in QueryMode],
        default=QueryMode.MIX.value,
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings(app_root=APP_ROOT)
    dataset_service = DatasetService(GraphRAGBenchLoader(), settings.active_dir)
    document = dataset_service.select_shortest(args.corpus, args.subset)

    factory = LightRAGFactory(settings)
    workspace_service = WorkspaceService(
        settings.workspace_dir,
        factory,
        factory.runtime_fingerprint(),
    )
    index_outcome = await workspace_service.ensure_ready(document)

    query_rag = await factory.create()
    try:
        result = await LightRAGAdapter().query(
            query_rag,
            args.query,
            QueryMode(args.mode),
            args.top_k,
        )
    finally:
        await query_rag.finalize_storages()

    return build_summary(document.corpus_name, index_outcome.action, result)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = asyncio.run(run(args))
    except AppError as exc:
        print(
            json.dumps(
                {
                    "status": "failure",
                    "error": {"code": exc.code, "message": exc.safe_message},
                },
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
                    "error": {
                        "code": "UNEXPECTED_ERROR",
                        "message": type(exc).__name__,
                    },
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
