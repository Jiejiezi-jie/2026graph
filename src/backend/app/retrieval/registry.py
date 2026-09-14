from app.domain.errors import AppError
from app.retrieval.contracts import Retriever


class RetrievalRegistry:
    def __init__(self, plugins: list[Retriever] | None = None):
        self._plugins: dict[str, Retriever] = {}
        for plugin in plugins or []:
            self.register(plugin)

    def register(self, plugin: Retriever) -> None:
        key = plugin.descriptor.id
        if key in self._plugins:
            raise ValueError(f"Duplicate retrieval method: {key}")
        self._plugins[key] = plugin

    def get(self, method_id: str) -> Retriever:
        if method_id not in self._plugins:
            raise AppError("UNKNOWN_RETRIEVAL_METHOD", "Unknown retrieval method")
        return self._plugins[method_id]

    def descriptors(self) -> list[dict]:
        return [p.descriptor.model_dump() for p in self._plugins.values()]
