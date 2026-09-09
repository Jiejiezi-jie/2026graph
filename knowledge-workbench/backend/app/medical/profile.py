from app.api.dependencies import AppContext
from app.domain.errors import AppError
from app.medical.bundle import MedicalBundle
from app.medical.engine import MedicalEngine
from app.medical.retrievers import AdaptiveRetriever, MedicalRetriever
from app.medical.preflight import runtime_issues
from app.retrieval.contracts import GraphData
from app.retrieval.registry import RetrievalRegistry
from app.services.graph_service import GraphService


class ImportedDataset:
    def __init__(self, bundle):
        self.bundle = bundle

    def load_active(self):
        return self.bundle.document

    def get_status(self):
        return {"selected": True, "dataset": self.bundle.document.model_dump(exclude={"context"})}


class ImportedWorkspace:
    imported = True

    def __init__(self, bundle, settings):
        self.bundle = bundle
        self.engine = MedicalEngine(bundle, settings)
        self.runtime_issues = runtime_issues(settings)
        shared_graph = GraphService(bundle.graph_dirs["lightrag"] / "graph_chunk_entity_relation.graphml")
        self.graphs = {"lightrag": shared_graph, "pathrag": shared_graph}

    def get_status(self, document):
        return {"status": "ready", "reusable": True, "corpus_name": self.bundle.document.corpus_name,
                "error_type": None}

    def build_registry(self):
        registry = RetrievalRegistry([MedicalRetriever(method, self.engine, self.bundle, self.graphs)
                                      for method in ("lightrag", "vector", "pathrag")])
        registry.register(AdaptiveRetriever(registry, self.engine.load_router))
        return registry

    def preview(self, method):
        if method in self.graphs:
            return self.graphs[method].full()
        if method in {"vector", "adaptive"}:
            message = ("Vector 使用独立文本向量索引，没有对应图谱。" if method == "vector"
                       else "Adaptive 查询后显示实际选中方法的图谱。")
            return GraphData(warnings=[message])
        raise AppError("UNKNOWN_RETRIEVAL_METHOD", "未知检索方法。")


def create_medical_context(settings):
    bundle = MedicalBundle(settings.medical_bundle)
    # Metadata reports the imported encoder, never the Novel embedding defaults.
    settings.embedding_model = "BAAI/bge-m3"
    settings.embedding_dim = 1024
    settings.embedding_max_token_size = 2048
    return AppContext(settings=settings, dataset_service=ImportedDataset(bundle),
                      workspace_service=ImportedWorkspace(bundle, settings))
