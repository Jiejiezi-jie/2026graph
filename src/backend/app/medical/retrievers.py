import asyncio
import json

from app.domain.errors import AppError
from app.retrieval.contracts import MethodDescriptor, MethodOption, RetrievalResult


def pathrag_evidence(context, parsed, chunk_store):
    # Context source numbers are not persistent chunk IDs. Match exact text only.
    by_content = {}
    for identifier, row in chunk_store.items():
        by_content.setdefault(row.get("content"), []).append(identifier)
    chunks = []
    for text in parsed["contexts"]:
        row = {"content": text}
        matches = by_content.get(text, [])
        if len(matches) == 1:
            row["chunk_id"] = matches[0]
        chunks.append(row)
    entities = []
    for row in parsed["entities"]:
        item = dict(row)
        if row.get("entity"):
            item["entity_name"] = row["entity"]
        entities.append(item)
    relationships = []
    for row in parsed["relations"]:
        item = dict(row)
        # Path prose is kept verbatim; it is never guessed into graph edges.
        if row.get("source") and row.get("target"):
            item.update(src_id=row["source"], tgt_id=row["target"])
        relationships.append(item)
    return RetrievalResult(context_text=context or None, entities=entities, relationships=relationships,
                           chunks=chunks, metadata={"context_origin": "pathrag_only_need_context",
                           "paths": parsed.get("paths", [])})


class MedicalRetriever:
    def __init__(self, method, engine, bundle, graphs):
        self.method, self.engine, self.bundle, self.graphs = method, engine, bundle, graphs
        descriptions = {
            "vector": "Medical 原文向量检索：使用队友的独立 BGE-M3 Chunk 索引。Top-K 为原文候选数。",
            "lightrag": "Medical 图谱检索：复用队友的 LightRAG 索引；图谱候选默认 40，Top-K 控制原文候选数。",
            "pathrag": "Medical 路径检索：在 LightRAG 构建的共享图谱和向量索引上执行 PathRAG 路径筛选。Top-K 控制检索候选数。",
        }
        options = []
        if method == "lightrag":
            options = [
                MethodOption(key="mode", label="检索模式", choices=["local", "global", "hybrid", "mix", "naive"], default="hybrid"),
                MethodOption(key="graph_top_k", label="图谱候选", choices=["5", "10", "20", "40", "50"], default="40"),
            ]
        self.descriptor = MethodDescriptor(id=method, name={"vector": "Vector", "lightrag": "LightRAG", "pathrag": "PathRAG"}[method],
                                           description=descriptions[method], options=options)

    async def retrieve(self, request):
        allowed = {option.key: option for option in self.descriptor.options}
        if set(request.options) - set(allowed) or any(
                value not in allowed[key].choices for key, value in request.options.items()):
            raise AppError("INVALID_RETRIEVAL_OPTIONS", "当前 Medical 方法不支持这些检索参数。")
        result = await self.engine.retrieve(self.method, request)
        if self.method != "vector":
            result.graph = await asyncio.to_thread(self.related_graph, result)
        result.metadata.update(corpus_name=self.bundle.document.corpus_name, subset="medical",
                               selected_method=self.method, index_source_commit=self.bundle.manifest["source_commit"],
                               corpus_sha256=self.bundle.corpus_hash, embedding_model="bge-m3",
                               embedding_pooling="cls", embedding_normalized=True)
        if self.method != "vector":
            result.metadata.update(graph_source="lightrag", graph_index="indexes/lightrag/medical",
                                   shared_graph=True)
        return result

    def related_graph(self, result):
        service = self.graphs[self.method]
        entities, relationships = result.entities, result.relationships
        if self.method == "pathrag":
            graph = service._load()
            def identifier(value):
                value = str(value or "")
                # Upstream CSV parsing removes one pair of literal double quotes.
                # Prefer an exact raw ID, then one exactly wrapped ID. No case/alias guessing.
                quoted = '"' + value + '"'
                unquoted = value[1:-1] if len(value) > 1 and value.startswith('"') and value.endswith('"') else value
                return value if value in graph else quoted if quoted in graph else unquoted if unquoted in graph else value
            entities = [{**row, "entity_name": identifier(row.get("entity_name"))} for row in entities]
            relationships = [{**row, "src_id": identifier(row.get("src_id")),
                               "tgt_id": identifier(row.get("tgt_id"))}
                              for row in relationships if row.get("src_id") and row.get("tgt_id")]
            # PathRAG also returns path evidence as prose, without src_id/tgt_id rows.
            # Highlight only explicit consecutive path nodes and edges present in the graph.
            for path in result.metadata.get("paths", []):
                if not isinstance(path, list) or not all(isinstance(node, str) for node in path):
                    continue
                resolved = [identifier(node) for node in path]
                entities.extend({"entity_name": node} for node in resolved if node in graph)
                relationships.extend({"src_id": source, "tgt_id": target}
                                     for source, target in zip(resolved, resolved[1:])
                                     if source != target and graph.has_edge(source, target))
        return service.related(entities, relationships)


class AdaptiveRetriever:
    descriptor = MethodDescriptor(
        id="adaptive", name="Adaptive",
        description="Medical 自适应路由：由英文 Medical 问题分类器选择 Vector、LightRAG 或 PathRAG，只执行选中的检索。Top-K 传给选中方法；LightRAG 图谱候选另设为 40。")

    def __init__(self, registry, predictor_loader, feature_encoder=None, router_kind="tfidf_logistic_regression"):
        self.registry, self.predictor_loader = registry, predictor_loader
        self.feature_encoder, self.router_kind = feature_encoder, router_kind

    async def route(self, request):
        if request.options:
            raise AppError("INVALID_RETRIEVAL_OPTIONS", "Adaptive 自动选择方法，不接受手动 mode 参数。")
        predictor = await asyncio.to_thread(self.predictor_loader)
        features = await self.feature_encoder(request.query) if self.feature_encoder else [request.query]
        selected = str((await asyncio.to_thread(predictor.predict, features))[0])
        if selected not in {"vector", "lightrag", "pathrag"}:
            raise AppError("INVALID_ROUTER", "路由模型返回了未知的检索方法。")
        probabilities = {}
        if hasattr(predictor, "predict_proba"):
            values = (await asyncio.to_thread(predictor.predict_proba, features))[0]
            classes = predictor.classes_
            probabilities = {str(label): float(value) for label, value in zip(classes, values)}
        # Keep the teammate's evaluated graph candidate budget; don't reduce it to UI default 5.
        options = {"mode": "hybrid", "graph_top_k": "40"} if selected == "lightrag" else {}
        routed = request.model_copy(update={"method_id": selected, "options": options})
        metadata = dict(selected_method=selected, routing_probabilities=probabilities,
                               router_kind=self.router_kind, effective_top_k=routed.top_k,
                               effective_options=options,
                               routing_note="路由概率是方法分类概率，不是答案正确率。训练域为英文 Medical。")
        return routed, metadata

    async def retrieve_routed(self, routed, metadata):
        result = await self.registry.get(routed.method_id).retrieve(routed)
        result.metadata.update(metadata)
        return result

    async def retrieve(self, request):
        routed, metadata = await self.route(request)
        return await self.retrieve_routed(routed, metadata)
