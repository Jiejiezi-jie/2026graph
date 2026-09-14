from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryMode(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"


class CorpusDocument(BaseModel):
    corpus_name: str
    context: str
    subset: str
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)


class QueryError(BaseModel):
    code: str
    message: str


class QueryResult(BaseModel):
    query: str
    mode: QueryMode
    answer: str | None = None
    latency_ms: float = Field(ge=0)
    context_text: str | None = None
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: QueryError | None = None
