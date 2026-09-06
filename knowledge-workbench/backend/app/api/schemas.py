from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    api_key_configured: bool
    llm_model: str
    embedding_model: str
    graph_storage: str


class DatasetSummary(BaseModel):
    corpus_name: str
    subset: str
    character_count: int
    word_count: int


class DatasetStatusResponse(BaseModel):
    selected: bool
    dataset: DatasetSummary | None = None


class IndexStatusResponse(BaseModel):
    status: Literal["not_built", "building", "ready", "failed", "interrupted"]
    reusable: bool
    corpus_name: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    latency_ms: float | None = None
    error_type: str | None = None
