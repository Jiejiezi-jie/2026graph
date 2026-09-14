from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


METHODS = ("vector", "lightrag", "pathrag")


@dataclass
class BackendResult:
    """Serializable result shared by all three real backends."""

    question_id: str
    method: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    entities: list[Any] = field(default_factory=list)
    relations: list[Any] = field(default_factory=list)
    paths: list[Any] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unsupported official method: {self.method}")
        if not isinstance(self.answer, str):
            raise TypeError("answer must be a string")
        for name in ("contexts", "entities", "relations", "paths"):
            if not isinstance(getattr(self, name), list):
                raise TypeError(f"{name} must be a list")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RAGBackend(ABC):
    """Async backend contract used by the official experiment runner."""

    method: str

    @abstractmethod
    async def index(self, corpus: str) -> dict[str, Any]:
        """Build or load an index and return measured indexing metadata."""

    @abstractmethod
    async def query(self, question: str, question_id: str = "") -> dict[str, Any]:
        """Retrieve evidence, generate an answer, and return BackendResult data."""

    async def close(self) -> None:
        """Release backend resources, if any."""

