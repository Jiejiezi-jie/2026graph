import asyncio
import hashlib
import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from app.config import Settings
from app.domain.errors import AppError
from app.retrieval.contracts import RetrievalRequest, RetrievalResult
from app.retrieval.registry import RetrievalRegistry
from app.services.query_service import QueryService
from app.services.run_store import RunStore


@pytest.fixture
def medical_bundle(tmp_path):
    root = tmp_path / "medical"
    corpus = "Medical evidence about alpha and beta."
    corpus_hash = hashlib.sha256(corpus.encode()).hexdigest()
    identity = {"embedding_model": "bge-m3", "embedding_dimension": 1024,
                "chunk_tokens": 1200, "chunk_overlap_tokens": 100}
    def write(path, data):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data), encoding="utf-8")
    for method, folder in [("lightrag", "lightrag/medical"), ("pathrag", "pathrag")]:
        write(f"indexes/{method}/official_index_identity.json",
              {"corpus_sha256": corpus_hash, "index_identity": identity})
        write(f"indexes/{method}/official_index_manifest.json",
              {"corpus_sha256": corpus_hash, "index_identity": identity})
        write(f"indexes/{folder}/kv_store_text_chunks.json",
              {"chunk-A": {"content": corpus}})
        write(f"indexes/{folder}/kv_store_full_docs.json",
              {"doc-A": {"content": corpus}})
        for name in ("chunks", "entities", "relationships"):
            # NanoVectorDB's serialized dimension is independent of model settings.
            write(f"indexes/{folder}/vdb_{name}.json",
                  {"embedding_dim": 1024, "data": [{"__id__": "a"}]})
        graph = nx.Graph()
        graph.add_edge("Alpha", "Beta", description="test relationship")
        nx.write_graphml(graph, root / f"indexes/{folder}/graph_chunk_entity_relation.graphml")
    write("indexes/vector/index.json", {"corpus_sha256": corpus_hash,
          "index_identity": identity, "chunks": [corpus, "Second evidence."]})
    np.save(root / "indexes/vector/vectors.npy", np.zeros((2, 1024), dtype="float32"))
    write("bundle.json", {"schema_version": 1, "source_commit": "abc123",
                         "corpus_name": "Medical", "embedding_max_length": 2048})
    write("router.joblib", {})
    return root


def test_medical_bundle_rejects_incomplete_or_incompatible_indexes(medical_bundle):
    from app.medical.bundle import MedicalBundle
    bundle = MedicalBundle(medical_bundle)
    assert bundle.document.context == "Medical evidence about alpha and beta."
    assert bundle.document.subset == "medical"
    assert bundle.summary()["vector_chunks"] == 2
    np.save(medical_bundle / "indexes/vector/vectors.npy", np.zeros((2, 384)))
    with pytest.raises(AppError, match="dimension|维度"):
        MedicalBundle(medical_bundle)


def test_medical_bundle_rejects_different_corpus(medical_bundle):
    from app.medical.bundle import MedicalBundle
    path = medical_bundle / "indexes/vector/index.json"
    data = json.loads(path.read_text())
    data["corpus_sha256"] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(AppError, match="corpus|原文"):
        MedicalBundle(medical_bundle)


def test_medical_bundle_rejects_missing_graph(medical_bundle):
    from app.medical.bundle import MedicalBundle
    (medical_bundle / "indexes/lightrag/medical/graph_chunk_entity_relation.graphml").unlink()
    with pytest.raises(AppError, match="graph|图谱|缺少"):
        MedicalBundle(medical_bundle)


def test_bundle_uses_only_shared_lightrag_documents(medical_bundle):
    from app.medical.bundle import MedicalBundle
    for folder in ("lightrag/medical", "pathrag"):
        path = medical_bundle / f"indexes/{folder}/kv_store_full_docs.json"
        path.write_text(json.dumps({"doc": {"content": "The same cleaned Medical corpus."}}))
    assert MedicalBundle(medical_bundle).document.context == "The same cleaned Medical corpus."
    path.write_text(json.dumps({"doc": {"content": "An unrelated corpus."}}))
    bundle = MedicalBundle(medical_bundle)
    assert bundle.document.context == "The same cleaned Medical corpus."
    assert bundle.graph_dirs["pathrag"] == bundle.graph_dirs["lightrag"]
    (medical_bundle / "indexes/pathrag/graph_chunk_entity_relation.graphml").unlink()
    assert MedicalBundle(medical_bundle).graph_dirs == bundle.graph_dirs


def test_shared_snapshot_copies_vectors_and_chunks_without_reembedding(medical_bundle):
    from app.medical.shared_index import FILES, prepare_pathrag_index
    source = medical_bundle / "indexes/lightrag/medical"
    cache = medical_bundle / "runtime/pathrag-shared"
    target = prepare_pathrag_index(source, cache)
    assert target != source
    assert all((target / name).read_bytes() == (source / name).read_bytes() for name in FILES)
    assert prepare_pathrag_index(source, cache) == target
    (source / "kv_store_text_chunks.json").write_text('{"changed": {"content": "new"}}')
    assert prepare_pathrag_index(source, cache) != target


def test_pathrag_evidence_keeps_only_exact_chunk_identifiers():
    from app.medical.retrievers import pathrag_evidence
    text = "-----Sources-----\n```csv\nid,content\n0,Exact evidence\n1,Unknown evidence\n```"
    parsed = {"contexts": ["Exact evidence", "Unknown evidence"], "entities": [
        {"entity": "Alpha", "type": "condition"}], "relations": [
        {"source": "Alpha", "target": "Beta", "description": "is linked"}], "paths": []}
    result = pathrag_evidence(text, parsed, {"c1": {"content": "Exact evidence"}})
    assert result.context_text == text
    assert result.chunks[0]["chunk_id"] == "c1"
    assert "chunk_id" not in result.chunks[1]
    assert result.entities[0]["entity_name"] == "Alpha"
    assert result.relationships[0]["src_id"] == "Alpha"
    assert result.relationships[0]["tgt_id"] == "Beta"


@pytest.mark.asyncio
async def test_pathrag_csv_names_resolve_to_quoted_graph_ids_without_fuzzy_matching(medical_bundle):
    from app.medical.bundle import MedicalBundle
    from app.medical.retrievers import MedicalRetriever
    from app.services.graph_service import GraphService
    graph = nx.Graph()
    graph.add_edge('"Alpha"', '"Beta"')
    path = medical_bundle / "indexes/pathrag/graph_chunk_entity_relation.graphml"
    nx.write_graphml(graph, path)
    class Engine:
        async def retrieve(self, method, request):
            return RetrievalResult(context_text="evidence",
                                   entities=[{"entity_name": "Alpha"}, {"entity_name": "Alph"}],
                                   relationships=[{"src_id": "Alpha", "tgt_id": "Beta"}])
    plugin = MedicalRetriever("pathrag", Engine(), MedicalBundle(medical_bundle),
                              {"pathrag": GraphService(path)})
    result = await plugin.retrieve(RetrievalRequest(query="test", method_id="pathrag"))
    assert {n.id for n in result.graph.nodes} == {'"Alpha"', '"Beta"'}
    assert result.graph.edges[0].retrieved is True
    assert result.entities[0]["entity_name"] == "Alpha"


def test_pathrag_highlights_verified_path_edges_without_inventing_shortcuts(medical_bundle):
    from app.medical.bundle import MedicalBundle
    from app.medical.retrievers import MedicalRetriever
    from app.services.graph_service import GraphService
    graph = nx.Graph()
    graph.add_edges_from([("Alpha", "Beta"), ("Beta", "Gamma"), ("Alpha", "Gamma")])
    path = medical_bundle / "indexes/lightrag/medical/graph_chunk_entity_relation.graphml"
    nx.write_graphml(graph, path)
    plugin = MedicalRetriever("pathrag", None, MedicalBundle(medical_bundle), {"pathrag": GraphService(path)})
    evidence = RetrievalResult(metadata={"paths": [["Alpha", "Beta", "Beta", "Gamma", "Unknown"]]})
    hits = plugin.related_graph(evidence)
    assert set(hits.hit_node_ids) == {"Alpha", "Beta", "Gamma"}
    assert {frozenset(pair) for pair in hits.hit_edge_pairs} == {frozenset(["Alpha", "Beta"]), frozenset(["Beta", "Gamma"])}
    assert evidence.entities == [] and evidence.relationships == []


@pytest.mark.asyncio
async def test_adaptive_dispatches_once_and_uses_shared_answer_service(tmp_path):
    from app.medical.retrievers import AdaptiveRetriever
    from app.retrieval.contracts import MethodDescriptor
    calls = []
    class Backend:
        descriptor = MethodDescriptor(id="pathrag", name="PathRAG", description="")
        async def retrieve(self, request):
            calls.append(request)
            return RetrievalResult(context_text="Path evidence", chunks=[{"content": "Path evidence"}])
    class Predictor:
        classes_ = np.array(["lightrag", "pathrag", "vector"])
        def predict(self, questions):
            return ["pathrag"]
        def predict_proba(self, questions):
            return np.array([[0.1, 0.7, 0.2]])
    class Generator:
        async def generate(self, query, result):
            return "Answer from " + result.context_text
    registry = RetrievalRegistry([Backend()])
    registry.register(AdaptiveRetriever(registry, lambda: Predictor()))
    service = QueryService(registry, Generator(), RunStore(tmp_path / "runs.db"), asyncio.Lock())
    response = await service.query(RetrievalRequest(query="Why?", method_id="adaptive", top_k=9))
    assert response["answer"] == "Answer from Path evidence"
    assert response["method_id"] == "adaptive"
    metadata = response["retrieval"]["metadata"]
    assert metadata["selected_method"] == "pathrag"
    assert metadata["routing_probabilities"]["pathrag"] == 0.7
    assert len(calls) == 1 and calls[0].method_id == "pathrag" and calls[0].top_k == 9
    assert service.runs.get(response["id"])["retrieval"]["metadata"] == metadata


@pytest.mark.asyncio
async def test_adaptive_unknown_label_does_not_silently_fallback():
    from app.medical.retrievers import AdaptiveRetriever
    predictor = SimpleNamespace(predict=lambda q: ["unknown"])
    with pytest.raises(AppError, match="路由|router"):
        await AdaptiveRetriever(RetrievalRegistry(), lambda: predictor).retrieve(
            RetrievalRequest(query="why", method_id="adaptive"))


def test_imported_profile_protects_data_and_exposes_method_specific_graph(medical_bundle, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.medical.profile import create_medical_context
    settings = Settings(app_root=tmp_path, medical_bundle=medical_bundle, _env_file=None)
    app = create_app(context=create_medical_context(settings))
    with TestClient(app) as client:
        assert client.get("/api/dataset/status").json()["dataset"]["subset"] == "medical"
        assert client.get("/api/health").json()["imported_profile"] is True
        assert client.get("/api/index/status").json()["reusable"] is True
        methods = client.get("/api/retrieval/methods").json()["methods"]
        assert {m["id"] for m in methods} == {"lightrag", "pathrag", "vector", "adaptive"}
        assert client.get("/api/graph?method_id=pathrag").json()["total_nodes"] == 2
        assert client.get("/api/graph?method_id=vector").json()["nodes"] == []
        assert client.get("/api/graph?method_id=unknown").status_code == 400
        assert client.post("/api/dataset/select", json={"subset": "novel"}).status_code == 409
        assert client.post("/api/index/build", json={"rebuild": True}).status_code == 409
        updated = client.post("/api/settings/connection", json={"base_url": "https://api.deepseek.com", "api_key": "unit-test"})
        assert updated.status_code == 200
        assert client.get("/api/index/status").json()["reusable"] is True
        assert "unit-test" not in client.get("/api/health").text


@pytest.mark.asyncio
async def test_vector_ranking_uses_imported_vectors_and_persistent_source_indices(medical_bundle, tmp_path):
    from app.medical.bundle import MedicalBundle
    from app.medical.engine import MedicalEngine
    vectors = np.zeros((2, 1024), dtype="float32")
    vectors[0, 1] = 1
    vectors[1, 0] = 1
    np.save(medical_bundle / "indexes/vector/vectors.npy", vectors)
    engine = MedicalEngine(MedicalBundle(medical_bundle), Settings(app_root=tmp_path, _env_file=None))
    class Embedding:
        async def embed(self, texts):
            return vectors[1:2]
    engine.embedding = Embedding()
    result = await engine.retrieve("vector", RetrievalRequest(query="test", top_k=1))
    assert result.chunks == [{"chunk_id": "vector:1", "source_index": 1, "content": "Second evidence.", "score": 1.0}]
    assert result.graph.nodes == []


@pytest.mark.asyncio
async def test_graph_bridge_requests_evidence_only_and_preserves_failure(medical_bundle, tmp_path):
    from app.medical.bundle import MedicalBundle
    from app.medical.engine import MedicalEngine
    engine = MedicalEngine(MedicalBundle(medical_bundle), Settings(app_root=tmp_path, _env_file=None))
    class Rag:
        async def aquery_data(self, query, param):
            assert param.top_k == 40 and param.chunk_top_k == 7 and param.mode == "hybrid"
            return {"status": "success", "data": {"entities": [], "relationships": [],
                    "chunks": [{"chunk_id": "c-real", "content": "real evidence"}], "references": []}}
        async def aquery(self, *args, **kwargs):
            pytest.fail("LightRAG bridge must not generate an answer")
    engine.backends["lightrag"] = SimpleNamespace(rag=Rag())
    result = await engine.retrieve("lightrag", RetrievalRequest(query="test", top_k=7))
    assert result.chunks[0]["chunk_id"] == "c-real"
    assert "real evidence" in result.context_text
    class Path:
        async def aquery(self, query, param):
            assert param.only_need_context is True and param.top_k == 7
            assert param.max_token_for_text_unit == 2000
            return "path context"
    engine.backends["pathrag"] = SimpleNamespace(rag=Path(), _query_param_cls=lambda **kwargs: SimpleNamespace(**kwargs))
    engine.module = lambda name: SimpleNamespace(parse_pathrag_context=lambda context: {
        "contexts": ["Medical evidence about alpha and beta."], "entities": [], "relations": [], "paths": []})
    result = await engine.retrieve("pathrag", RetrievalRequest(query="test", top_k=7))
    assert result.context_text == "path context"
    assert result.chunks[0]["chunk_id"] == "chunk-A"


def test_preflight_explains_missing_model_without_loading_it(tmp_path):
    from app.medical.preflight import runtime_issues
    settings = Settings(app_root=tmp_path, medical_embedding_path=tmp_path / "missing", _env_file=None)
    issues = runtime_issues(settings)
    assert any("BGE-M3" in issue for issue in issues)
    assert any("PathRAG" in issue for issue in issues)
