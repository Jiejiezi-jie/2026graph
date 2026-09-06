import asyncio
import json

from app.domain.errors import AppError
from app.retrieval.contracts import MethodDescriptor, MethodOption, RetrievalRequest, RetrievalResult
from app.services.web_errors import safe_error
from app.services.rag_lifecycle import finalize_rag


class LightRAGRetriever:
    descriptor = MethodDescriptor(
        id="lightrag", name="LightRAG", description="复用现有知识图谱与向量索引，支持图谱检索与纯向量检索。",
        options=[MethodOption(key="mode", label="检索模式", choices=["local", "global", "hybrid", "mix", "naive"], default="mix")],
    )

    def __init__(self, dataset, workspace, graph):
        self.dataset, self.workspace, self.graph = dataset, workspace, graph

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        from lightrag import QueryParam

        mode = request.options.get("mode", "mix")
        if mode not in self.descriptor.options[0].choices:
            raise AppError("INVALID_MODE", "请选择 local / global / hybrid / mix / vector（naive）。")
        if set(request.options) - {"mode"}:
            raise AppError("INVALID_OPTIONS", "LightRAG 收到了不支持的检索参数。")
        document = self.dataset.load_active()
        if document is None or not self.workspace.get_status(document)["reusable"]:
            raise AppError("INDEX_NOT_READY", "当前数据的索引尚未就绪或配置不匹配；查询不会自动建图。")
        rag = None
        try:
            rag = await self.workspace.factory.create()
            raw = await rag.aquery_data(
                request.query,
                param=QueryParam(mode=mode, top_k=request.top_k, chunk_top_k=request.top_k,
                                 stream=False, enable_rerank=False),
            )
        except Exception as exc:
            raise safe_error(exc) from exc
        finally:
            if rag is not None:
                # On success, persistence failures must also fail the request.
                # On failure preserve the primary exception rather than masking it.
                import sys
                failing = sys.exc_info()[0] is not None
                try:
                    await finalize_rag(rag)
                except Exception as exc:
                    if not failing:
                        raise safe_error(exc) from exc
        if not isinstance(raw, dict) or raw.get("status") != "success":
            metadata = raw.get("metadata") if isinstance(raw, dict) else {}
            if isinstance(metadata, dict) and metadata.get("failure_reason") == "no_results":
                return RetrievalResult(warnings=["没有检索到相关证据，未调用答案生成。"])
            raise AppError("LIGHTRAG_QUERY_FAILED", "LightRAG 未返回成功结果，请检查索引及模型连接。")
        data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
        fields, warnings = {}, []
        for key in ("entities", "relationships", "chunks", "references"):
            value = data.get(key)
            fields[key] = [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
            if not isinstance(value, list):
                warnings.append(f"LightRAG 未提供 {key}，按空列表展示。")
        context = "\n\n".join(
            "## " + key + "\n" + json.dumps(value, ensure_ascii=False, default=str)
            for key, value in fields.items() if value
        ) or None
        result = RetrievalResult(**fields, context_text=context, warnings=warnings,
                                 metadata={"mode": mode, "corpus_name": getattr(document, "corpus_name", None),
                                           "context_origin": "adapter_serialization"})
        if mode != "naive":
            try:
                result.graph = await asyncio.to_thread(self.graph.related, result.entities, result.relationships)
            except AppError as exc:
                result.warnings.append(exc.safe_message)
        return result
