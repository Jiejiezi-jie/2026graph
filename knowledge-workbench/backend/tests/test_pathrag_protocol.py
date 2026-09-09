import asyncio
import json
from types import SimpleNamespace

import pytest

from app.domain.errors import AppError


KEYWORDS = {
    "high_level_keywords": ["Anatomical layers", "Tumor penetration"],
    "low_level_keywords": ["Mucosa", "Serosa", "Bile duct wall"],
}


def test_recovers_exact_double_brace_output_without_altering_keyword_text():
    from app.medical.pathrag_protocol import normalize_keywords
    data = {**KEYWORDS, "low_level_keywords": ["Literal {detail}", "Serosa"]}
    raw = "{" + json.dumps(data) + "}"
    assert json.loads(normalize_keywords(raw)) == data
    assert json.loads(normalize_keywords(chr(96) * 3 + "json\n" + raw + "\n" + chr(96) * 3)) == data
    assert json.loads(normalize_keywords(json.dumps(data))) == data


def test_combined_sources_preserve_commas_and_multiline_chunk_boundaries():
    from app.medical.pathrag_protocol import combined_source_texts
    first = "First paragraph, with commas.\nSecond paragraph, still the same chunk."
    second = "Another chunk, containing evidence."
    fence = chr(96) * 3
    context = "-----Sources-----\n" + fence + "csv\nid,\tcontent\n1,\t" + first + "\n2,\t" + second + "\n" + fence
    assert combined_source_texts(context) == [first, second]
    # Normal quoted CSV is delegated to the existing vendor CSV parser.
    quoted = "-----Sources-----\n" + fence + 'csv\nid,content\n0,"a,b"\n' + fence
    assert combined_source_texts(quoted) is None


@pytest.mark.parametrize("raw, code", [
    ('{"high_level_keywords": ["x"],', "PATHRAG_KEYWORDS_INVALID"),
    ('{"high_level_keywords": "x", "low_level_keywords": ["y"]}', "PATHRAG_KEYWORDS_INVALID"),
    ('{"high_level_keywords": [], "low_level_keywords": ["y"]}', "PATHRAG_KEYWORDS_EMPTY"),
    ('{"high_level_keywords": ["x"], "low_level_keywords": [42]}', "PATHRAG_KEYWORDS_INVALID"),
])
def test_invalid_keywords_raise_actionable_errors_instead_of_empty_evidence(raw, code):
    from app.medical.pathrag_protocol import normalize_keywords
    with pytest.raises(AppError) as error:
        normalize_keywords(raw)
    assert error.value.code == code


@pytest.mark.asyncio
async def test_session_llm_repairs_examples_requests_json_and_normalizes_response(monkeypatch):
    from app.config import Settings
    from app.medical.engine import SessionLLM
    calls = []
    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="{" + json.dumps(KEYWORDS) + "}"))])
    monkeypatch.setattr("openai.AsyncOpenAI", FakeClient)
    settings = Settings(deepseek_api_key="unit-test", _env_file=None)
    llm = SessionLLM(settings, pathrag_keywords=True)
    example = 'Example:\n{{"high_level_keywords":["x"],"low_level_keywords":["y"]}}'
    llm.keyword_examples = (example,)
    raw = await llm.complete(example + '\nQuery: literal {{braces}}', keyword_extraction=True)
    assert json.loads(raw) == KEYWORDS
    assert calls[0]["response_format"] == {"type": "json_object"}
    prompt = calls[0]["messages"][-1]["content"]
    assert 'Example:\n{"high_level_keywords":["x"],"low_level_keywords":["y"]}' in prompt
    assert 'Query: literal {{braces}}' in prompt
    await SessionLLM(settings).complete("ordinary prompt", keyword_extraction=True)
    assert "response_format" not in calls[1]


@pytest.mark.asyncio
async def test_pathrag_failure_is_saved_as_failure_without_answer_generation(tmp_path):
    from app.config import Settings
    from app.medical.engine import MedicalEngine
    from app.retrieval.contracts import MethodDescriptor, RetrievalRequest
    from app.retrieval.registry import RetrievalRegistry
    from app.services.query_service import QueryService
    from app.services.run_store import RunStore
    engine = MedicalEngine(SimpleNamespace(manifest={"source_commit":"test"}), Settings(_env_file=None))
    class Rag:
        async def aquery(self, *args):
            return "Sorry, I'm not able to provide an answer to that question."
    engine.backends["pathrag"] = SimpleNamespace(rag=Rag(), _query_param_cls=lambda **kw: kw)
    engine.module = lambda _: SimpleNamespace(parse_pathrag_context=lambda _: {
        "contexts": [], "entities": [], "relations": [], "paths": []})
    class Retriever:
        descriptor = MethodDescriptor(id="pathrag", name="PathRAG", description="")
        async def retrieve(self, request):
            return await engine.retrieve("pathrag", request)
    class Generator:
        async def generate(self, *args):
            pytest.fail("Do not generate an answer for failed retrieval")
    store = RunStore(tmp_path / "runs.db")
    service = QueryService(RetrievalRegistry([Retriever()]), Generator(), store, asyncio.Lock())
    response = await service.query(RetrievalRequest(query="bile duct", method_id="pathrag"))
    assert response["status"] == "failure"
    assert response["error"]["code"] == "PATHRAG_RETRIEVAL_FAILED"
    assert store.get(response["id"])["status"] == "failure"
