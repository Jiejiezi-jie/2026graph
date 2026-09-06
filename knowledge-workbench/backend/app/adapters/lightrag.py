import asyncio
import json
import time
from typing import Any

from lightrag import QueryParam

from app.domain.errors import AppError, LightRAGQueryError
from app.domain.models import QueryMode, QueryResult


STRUCTURED_FIELDS = ("entities", "relationships", "chunks", "references")
SECTION_LABELS = {
    "entities": "Entities",
    "relationships": "Relationships",
    "chunks": "Chunks",
    "references": "References",
}


def serialize_context(data: dict[str, Any]) -> str | None:
    """Serialize retrieval data for display, not as an exact internal prompt."""

    sections: list[str] = []
    for field in STRUCTURED_FIELDS:
        values = data.get(field)
        if not isinstance(values, list) or not values:
            continue
        rendered = json.dumps(values, ensure_ascii=False, indent=2, default=str)
        sections.append(f"## {SECTION_LABELS[field]}\n{rendered}")
    return "\n\n".join(sections) or None


def map_lightrag_exception(exc: Exception) -> AppError:
    name = type(exc).__name__.lower()
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name:
        return AppError("LLM_TIMEOUT", "The language model request timed out")
    if "authentication" in name or "permission" in name:
        return AppError("LLM_AUTH_FAILED", "The language model credentials were rejected")
    if "ratelimit" in name or "rate_limit" in name:
        return AppError("LLM_RATE_LIMITED", "The language model rate limit was reached")
    return AppError("LIGHTRAG_ERROR", "LightRAG could not complete the query")


class LightRAGAdapter:
    async def query(
        self,
        rag: Any,
        query: str,
        mode: QueryMode,
        top_k: int,
        system_prompt: str | None = None,
    ) -> QueryResult:
        if not isinstance(query, str) or not query.strip():
            raise AppError("INVALID_QUERY", "The query must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise AppError("INVALID_QUERY", "Top-K must be between 1 and 50")

        started = time.perf_counter()
        try:
            call_kwargs: dict[str, Any] = {
                "param": QueryParam(
                    mode=mode.value,
                    top_k=top_k,
                    chunk_top_k=top_k,
                    stream=False,
                )
            }
            if system_prompt is not None:
                call_kwargs["system_prompt"] = system_prompt
            result = await rag.aquery_llm(
                query,
                **call_kwargs,
            )
        except Exception as exc:
            raise map_lightrag_exception(exc) from exc

        if not isinstance(result, dict):
            raise LightRAGQueryError("LightRAG returned an invalid result")
        if result.get("status") != "success":
            message = result.get("message", "LightRAG query failed")
            if not isinstance(message, str) or not message:
                message = "LightRAG query failed"
            raise LightRAGQueryError(message)

        raw_data = result.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        warnings: list[str] = []
        structured: dict[str, list[dict[str, Any]]] = {}
        for field in STRUCTURED_FIELDS:
            value = data.get(field)
            if not isinstance(value, list):
                structured[field] = []
                warnings.append(f"LightRAG did not return a valid {field} list")
            else:
                structured[field] = value

        raw_llm_response = result.get("llm_response")
        llm_response = raw_llm_response if isinstance(raw_llm_response, dict) else {}
        answer = llm_response.get("content")
        if not isinstance(answer, str) or not answer.strip():
            answer = None
            warnings.append("LightRAG did not return answer content")

        normalized_data = {**data, **structured}
        return QueryResult(
            query=query,
            mode=mode,
            answer=answer,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            context_text=serialize_context(normalized_data),
            entities=structured["entities"],
            relationships=structured["relationships"],
            chunks=structured["chunks"],
            references=structured["references"],
            warnings=warnings,
        )
