import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4

from app.domain.errors import AppError
from app.retrieval.contracts import RetrievalRequest, RetrievalResult
from app.services.web_errors import safe_error


class QueryService:
    def __init__(self, registry, generator, runs, lock: asyncio.Lock):
        self.registry, self.generator, self.runs, self.lock = registry, generator, runs, lock

    async def query(self, request: RetrievalRequest) -> dict:
        retriever = self.registry.get(request.method_id)
        if self.lock.locked():
            raise AppError("WORKSPACE_BUSY", "当前正在建索引或查询，请等待完成。")
        async with self.lock:
            started = time.perf_counter()
            result = RetrievalResult()
            response = {
                "id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
                "query": request.query, "method_id": request.method_id, "options": request.options,
                "top_k": request.top_k, "status": "success", "answer": None, "error": None,
                "retrieval_ms": 0.0, "generation_ms": 0.0,
            }
            phase = "retrieval"
            phase_start = started
            try:
                result = await retriever.retrieve(request)
                response["retrieval_ms"] = round((time.perf_counter() - phase_start) * 1000, 2)
                phase = "generation"
                phase_start = time.perf_counter()
                if result.context_text:
                    response["answer"] = await self.generator.generate(request.query, result)
                else:
                    response["answer"] = "没有足够的检索证据来回答此问题。"
                response["generation_ms"] = round((time.perf_counter() - phase_start) * 1000, 2)
            except Exception as exc:
                error = safe_error(exc)
                response["status"] = "failure"
                response["error"] = {"code": error.code, "message": error.safe_message}
                response[phase + "_ms"] = round((time.perf_counter() - phase_start) * 1000, 2)
            response["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            response["retrieval"] = result.model_dump(mode="json")
            try:
                await asyncio.to_thread(self.runs.save, response)
            except Exception:
                response["retrieval"]["warnings"].append("本次查询记录未能保存到本地。")
            return response
