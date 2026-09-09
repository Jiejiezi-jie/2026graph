from app.retrieval.lightrag import LightRAGRetriever
from app.retrieval.registry import RetrievalRegistry


def build_registry(context, graph) -> RetrievalRegistry:
    registry = RetrievalRegistry()
    registry.register(LightRAGRetriever(context.dataset_service, context.workspace_service, graph))
    # Add custom implementations here: registry.register(MyRetriever(...)).
    # HTTP handlers, answer generation and the frontend do not need changing.
    return registry
