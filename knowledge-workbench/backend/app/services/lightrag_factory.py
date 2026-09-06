import hashlib
import json
from functools import partial
from typing import Any

from app.config import Settings
from app.domain.errors import AppError


PINNED_LIGHTRAG_COMMIT = "c1248646e4eda4d89054926af2e094730daf23fe"


class LightRAGFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tokenizer: Any | None = None
        self._embedding_model: Any | None = None

    async def create(self) -> Any:
        api_key = self._api_key()
        embedding_func = self._create_embedding_func()

        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache

        llm_func = partial(
            openai_complete_if_cache,
            self.settings.llm_model,
            base_url=self.settings.llm_base_url,
            api_key=api_key,
        )
        try:
            rag = LightRAG(
                working_dir=str(self.settings.workspace_dir),
                llm_model_func=llm_func,
                llm_model_name=self.settings.llm_model,
                embedding_func=embedding_func,
                chunk_token_size=self.settings.chunk_token_size,
                chunk_overlap_token_size=self.settings.chunk_overlap_token_size,
                graph_storage=self.settings.graph_storage,
            )
            await rag.initialize_storages()
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                "LIGHTRAG_INITIALIZATION_FAILED",
                "LightRAG could not initialize its storages",
            ) from exc
        return rag

    def runtime_fingerprint(self) -> str:
        payload = {
            "lightrag_commit": PINNED_LIGHTRAG_COMMIT,
            "llm_base_url": self.settings.llm_base_url,
            "llm_model": self.settings.llm_model,
            "embedding_model": self.settings.embedding_model,
            "embedding_dim": self.settings.embedding_dim,
            "embedding_max_token_size": self.settings.embedding_max_token_size,
            "chunk_token_size": self.settings.chunk_token_size,
            "chunk_overlap_token_size": self.settings.chunk_overlap_token_size,
            "graph_storage": self.settings.graph_storage,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def reset_shared_storage() -> None:
        """Only call with no active RAG and under the application's mutation gate.

        finalize_storages() does not clear namespace caches in the pinned
        LightRAG version. A new constructor initializes shared data again.
        """
        from lightrag.kg.shared_storage import finalize_share_data
        finalize_share_data()

    def _api_key(self) -> str:
        if self.settings.deepseek_api_key is None:
            raise AppError("MISSING_API_KEY", "DEEPSEEK_API_KEY is not configured")
        value = self.settings.deepseek_api_key.get_secret_value().strip()
        if not value:
            raise AppError("MISSING_API_KEY", "DEEPSEEK_API_KEY is not configured")
        return value

    def _create_embedding_func(self) -> Any:
        try:
            from lightrag.llm.hf import hf_embed
            from lightrag.utils import EmbeddingFunc
            from transformers import AutoModel, AutoTokenizer

            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.embedding_model
                )
            if self._embedding_model is None:
                self._embedding_model = AutoModel.from_pretrained(
                    self.settings.embedding_model
                )
            return EmbeddingFunc(
                embedding_dim=self.settings.embedding_dim,
                max_token_size=self.settings.embedding_max_token_size,
                model_name=self.settings.embedding_model,
                func=partial(
                    hf_embed.func,
                    tokenizer=self._tokenizer,
                    embed_model=self._embedding_model,
                ),
            )
        except Exception as exc:
            raise AppError(
                "EMBEDDING_ERROR",
                "The local embedding model could not be loaded",
            ) from exc
