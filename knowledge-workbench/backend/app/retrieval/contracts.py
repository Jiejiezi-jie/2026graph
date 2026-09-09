from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MethodOption(BaseModel):
    key: str
    label: str
    choices: list[str]
    default: str


class MethodDescriptor(BaseModel):
    id: str
    name: str
    description: str
    options: list[MethodOption] = Field(default_factory=list)


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    method_id: str = Field(default="lightrag", max_length=80)
    top_k: int = Field(default=5, ge=1, le=50, strict=True)
    options: dict[str, str] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be blank")
        return value.strip()


class GraphNode(BaseModel):
    id: str
    label: str
    type: str = "UNKNOWN"
    description: str = ""
    retrieved: bool = False
    position: dict[str, float] | None = None


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str = ""
    description: str = ""
    retrieved: bool = False
    directed: bool = False


class GraphData(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    truncated: bool = False
    total_nodes: int = 0
    total_edges: int = 0
    warnings: list[str] = Field(default_factory=list)
    layout_key: str | None = None
    hit_node_ids: list[str] = Field(default_factory=list)
    hit_edge_pairs: list[list[str]] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Evidence only. Missing scores/ranks/references are never synthesized."""

    context_text: str | None = None
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    graph: GraphData = Field(default_factory=GraphData)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Retriever(Protocol):
    descriptor: MethodDescriptor

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...


class AnswerGenerator(Protocol):
    async def generate(self, query: str, result: RetrievalResult) -> str: ...
