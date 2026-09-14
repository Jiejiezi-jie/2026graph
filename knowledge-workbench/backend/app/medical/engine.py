"""Retrieval-only bridge to a pinned teammate snapshot. No index() or ainsert()."""
import asyncio
import hashlib
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import warnings

import numpy as np

from app.domain.errors import AppError
from app.medical.retrievers import pathrag_evidence
from app.medical.shared_index import prepare_pathrag_index
from app.medical.pathrag_protocol import FAIL_RESPONSE, combined_source_texts, keyword_prompt, normalize_keywords
from app.retrieval.contracts import RetrievalResult


class SessionLLM:
    def __init__(self, settings, *, pathrag_keywords=False):
        self.settings = settings
        self.pathrag_keywords = pathrag_keywords
        self.keyword_examples = ()

    @property
    def model_name(self):
        return self.settings.llm_model

    async def complete(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        from openai import AsyncOpenAI
        secret = self.settings.deepseek_api_key
        if not secret or not secret.get_secret_value().strip():
            raise AppError("MISSING_API_KEY", "请先在网页 API 设置中填写 Key。")
        keyword_request = self.pathrag_keywords and kwargs.get("keyword_extraction", False)
        if keyword_request:
            prompt = keyword_prompt(prompt, self.keyword_examples)
        response_format = kwargs.get("response_format")
        if keyword_request:
            response_format = {"type": "json_object"}
        structured_keywords = keyword_request or (isinstance(response_format, dict) and response_format.get("type") == "json_object")
        budget = int(kwargs.get("max_tokens") or self.settings.medical_keyword_max_tokens)
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        messages += list(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        async with AsyncOpenAI(api_key=secret.get_secret_value(), base_url=self.settings.llm_base_url,
                               timeout=120, max_retries=1) as client:
            response = await client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=0,
                max_tokens=budget,
                **({"response_format": response_format} if response_format else {}))
        if not response.choices:
            raise AppError("EMPTY_LLM_RESPONSE", "检索关键词模型未返回候选响应。")
        choice = response.choices[0]
        content = choice.message.content
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        logging.getLogger(__name__).info(
            "retrieval_llm model=%s max_tokens=%s finish_reason=%s prompt_tokens=%s completion_tokens=%s reasoning_tokens=%s content_chars=%s reasoning_chars=%s",
            self.model_name, budget, choice.finish_reason,
            getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
            getattr(details, "reasoning_tokens", None), len(content or ""),
            len(getattr(choice.message, "reasoning_content", None) or ""))
        if choice.finish_reason == "length":
            code = "PATHRAG_KEYWORDS_TRUNCATED" if keyword_request else "LLM_KEYWORDS_TRUNCATED"
            raise AppError(code, "关键词响应达到输出长度上限，尚未开始图谱召回。请检查关键词输出预算。")
        if not content or not content.strip():
            raise AppError("EMPTY_LLM_RESPONSE", "检索关键词模型未返回有效内容。")
        if structured_keywords and not keyword_request:
            try:
                parsed = json.loads(content)
                if not isinstance(parsed, dict) or any(
                    not isinstance(parsed.get(key), list) or any(not isinstance(item, str) for item in parsed[key])
                    for key in ("high_level_keywords", "low_level_keywords")
                ):
                    raise ValueError("Invalid keyword structure")
            except (ValueError, TypeError) as exc:
                raise AppError("LLM_KEYWORDS_INVALID", "关键词模型未返回有效的关键词 JSON，尚未开始图谱召回。") from exc
        return normalize_keywords(content) if keyword_request else content


class MedicalEngine:
    def __init__(self, bundle, settings):
        self.bundle, self.settings = bundle, settings
        self.embedding = None
        self.backends = {}
        self.predictor = None
        self.package_name = "_medical_vendor_" + bundle.manifest["source_commit"][:12]

    def module(self, name):
        if self.package_name not in sys.modules:
            directory = self.bundle.root / "vendor/official_backends"
            self.bundle.require(directory / "__init__.py")
            # Verify executable imported snapshot files against the export manifest.
            for path in directory.glob("*.py"):
                relative = path.relative_to(self.bundle.root).as_posix()
                expected = self.bundle.manifest.get("files", {}).get(relative, {}).get("sha256")
                if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise AppError("INVALID_MEDICAL_BUNDLE", "Medical 适配器源码与导入清单不一致，请重新导入。")
            spec = importlib.util.spec_from_file_location(self.package_name, directory / "__init__.py",
                                                        submodule_search_locations=[str(directory)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[self.package_name] = module
            spec.loader.exec_module(module)
        return importlib.import_module(self.package_name + "." + name)

    def load_router(self):
        if self.predictor is None:
            try:
                import joblib
                from sklearn.exceptions import InconsistentVersionWarning
            except ImportError as exc:
                raise AppError("MEDICAL_DEPENDENCIES_MISSING", "请安装 Medical 独立环境中的 joblib 和 scikit-learn。") from exc
            artifact = self.bundle.manifest.get("router", {}).get("artifact", "router.joblib")
            path = self.bundle.root / artifact
            if not path.resolve().is_relative_to(self.bundle.root.resolve()):
                raise AppError("INVALID_ROUTER", "路由模型路径不合法。")
            expected = self.bundle.manifest.get("files", {}).get(artifact, {}).get("sha256")
            if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise AppError("INVALID_ROUTER", "路由模型与导入清单不一致，请重新导入。")
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                try:
                    self.predictor = joblib.load(path)
                except Exception as exc:
                    raise AppError("INVALID_ROUTER", "路由模型无法加载，请使用与训练相同的 scikit-learn 版本。") from exc
            if set(self.predictor.classes_) != {"vector", "lightrag", "pathrag"}:
                self.predictor = None
                raise AppError("INVALID_ROUTER", "路由模型类别与 Medical 检索器不匹配。")
        return self.predictor

    async def get_embedding(self):
        if self.embedding is None:
            path = self.settings.medical_embedding_path
            if not path or not (path / "config.json").is_file():
                raise AppError("MEDICAL_MODEL_MISSING", "未找到本地 BGE-M3，请下载后通过 --medical-embedding-path 指定目录。")
            client = self.module("model_client").TransformersEmbeddingClient(
                model_path=path, device=self.settings.medical_device, batch_size=4,
                max_length=2048, normalize=True)
            dimension = await asyncio.to_thread(lambda: client.dimension)
            if dimension != 1024:
                client.unload()
                raise AppError("EMBEDDING_MISMATCH", "Medical 查询模型必须是 1024 维 BGE-M3。")
            self.embedding = client
        return self.embedding

    @property
    def router_kind(self):
        return self.bundle.manifest.get("router", {}).get("kind", "tfidf_logistic_regression")

    async def router_features(self, question):
        if self.router_kind == "tfidf_logistic_regression":
            return [question]
        if self.router_kind != "bge_m3_cls_logistic_regression":
            raise AppError("INVALID_ROUTER", "不支持的路由特征配置。")
        client = await self.get_embedding()
        # Reuse retrieval weights, but preserve the router's training cutoff (512).
        # The imported client's lock also serializes retrieval inference.
        def encode():
            import torch
            import torch.nn.functional as functional
            with client._lock:
                client._load()
                encoded = client._tokenizer([question], padding=True, truncation=True,
                                            max_length=512, return_tensors="pt")
                encoded = {key: value.to(client.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    hidden = client._model(**encoded).last_hidden_state[:, 0]
                    vectors = functional.normalize(hidden, p=2, dim=1)
                return vectors.float().cpu().numpy()
        features = await asyncio.to_thread(encode)
        if features.shape != (1, 1024) or not np.isfinite(features).all():
            raise AppError("INVALID_ROUTER", "路由问题向量必须为有效的 1024 维向量。")
        return features

    async def graph_backend(self, method):
        if method not in self.backends:
            embedding = await self.get_embedding()
            llm = SessionLLM(self.settings, pathrag_keywords=method == "pathrag")
            if method == "lightrag":
                backend = self.module("lightrag_backend").LightRAGBackend(
                    self.bundle.root / "indexes/lightrag", llm, embedding)
            else:
                upstream = self.settings.medical_pathrag_root
                if not upstream or not (upstream / "PathRAG/__init__.py").is_file():
                    raise AppError("MEDICAL_DEPENDENCIES_MISSING", "未找到指定版本 PathRAG，请设置 --medical-pathrag-root。")
                backend = self.module("pathrag_backend").PathRAGBackend(
                    upstream, await asyncio.to_thread(prepare_pathrag_index,
                        self.bundle.graph_dirs["lightrag"], self.bundle.root / "runtime/pathrag-shared"),
                    llm, embedding)
            try:
                await backend._initialize()
                if method == "pathrag":
                    from PathRAG.prompt import PROMPTS
                    llm.keyword_examples = tuple(PROMPTS["keywords_extraction_examples"])
            except Exception:
                await backend.close()
                raise
            self.backends[method] = backend
        return self.backends[method]

    async def retrieve(self, method, request):
        if method == "vector":
            embedding = await self.get_embedding()
            metadata = self.bundle.read("indexes/vector/index.json")
            vectors = np.load(self.bundle.root / "indexes/vector/vectors.npy", allow_pickle=False)
            query = (await embedding.embed([request.query]))[0]
            scores = vectors @ query
            indices = np.argsort(-scores, kind="stable")[:request.top_k]
            chunks = [{"chunk_id": f"vector:{int(i)}", "source_index": int(i),
                       "content": metadata["chunks"][int(i)], "score": float(scores[i])} for i in indices]
            return RetrievalResult(chunks=chunks, context_text=json.dumps(chunks, ensure_ascii=False),
                                   metadata={"context_origin": "vector_chunks"})
        backend = await self.graph_backend(method)
        if method == "lightrag":
            from lightrag import QueryParam
            mode = request.options.get("mode", "hybrid")
            graph_top_k = int(request.options.get("graph_top_k", "40"))
            raw = await backend.rag.aquery_data(request.query, param=QueryParam(
                mode=mode, top_k=graph_top_k, chunk_top_k=request.top_k,
                max_entity_tokens=1500, max_relation_tokens=1500, max_total_tokens=12000,
                enable_rerank=False))
            if not isinstance(raw, dict) or raw.get("status") not in {"success", None}:
                if isinstance(raw, dict) and raw.get("status") == "failure":
                    raise AppError("MEDICAL_RETRIEVAL_FAILED", "LightRAG 检索未成功返回证据。")
                raise AppError("MEDICAL_RETRIEVAL_FAILED", "LightRAG 返回格式不正确。")
            data = raw.get("data") or {}
            fields = {key: data.get(key) or [] for key in ("entities", "relationships", "chunks", "references")}
            context = json.dumps(fields, ensure_ascii=False) if any(fields.values()) else None
            return RetrievalResult(**fields, context_text=context, metadata={
                "context_origin": "adapter_serialization", "mode": mode,
                "graph_top_k": graph_top_k, "chunk_top_k": request.top_k})
        context = await backend.rag.aquery(request.query, backend._query_param_cls(
            mode="hybrid", only_need_context=True, top_k=request.top_k,
            max_token_for_text_unit=2000, max_token_for_global_context=1500,
            max_token_for_local_context=1500))
        context = str(context or "")
        if context.strip() == FAIL_RESPONSE:
            raise AppError("PATHRAG_RETRIEVAL_FAILED",
                           "PathRAG 在生成检索上下文之前返回了失败提示，不能据此判断知识库没有证据。")
        parsed = self.module("pathrag_backend").parse_pathrag_context(context)
        sources = combined_source_texts(context)
        if sources is not None:
            parsed["contexts"] = sources
        if not any(parsed[key] for key in ("contexts", "entities", "relations")):
            return RetrievalResult(warnings=["PathRAG 未返回可解析的检索证据。"])
        return pathrag_evidence(context, parsed, self.bundle.read(self.bundle.graph_dirs["lightrag"] / "kv_store_text_chunks.json"))

    async def close(self):
        for backend in self.backends.values():
            await backend.close()
        self.backends.clear()
        if self.embedding is not None:
            self.embedding.unload()
            self.embedding = None
