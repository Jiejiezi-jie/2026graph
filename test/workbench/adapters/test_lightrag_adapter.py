import pytest

from app.adapters.lightrag import LightRAGAdapter
from app.domain.errors import AppError, LightRAGQueryError
from app.domain.models import QueryMode


class SuccessfulFakeRAG:
    def __init__(self) -> None:
        self.param = None

    async def aquery_llm(self, query, param):
        self.param = param
        return {
            "status": "success",
            "data": {
                "entities": [{"entity_name": "ALICE"}],
                "relationships": [],
                "chunks": [{"content": "Alice appears."}],
                "references": [],
            },
            "llm_response": {"content": "Alice is the protagonist."},
        }


class RaisingFakeRAG:
    async def aquery_llm(self, query, param):
        raise TimeoutError("provider timed out with private details")


class FailureFakeRAG:
    async def aquery_llm(self, query, param):
        return {
            "status": "failure",
            "message": "no results",
            "data": {},
            "llm_response": {"content": None},
        }


class MissingDataFakeRAG:
    async def aquery_llm(self, query, param):
        return {
            "status": "success",
            "data": {"chunks": [{"content": "Only a chunk."}]},
            "llm_response": {"content": "An answer."},
        }


class SystemPromptFakeRAG:
    def __init__(self) -> None:
        self.system_prompt = None

    async def aquery_llm(self, query, param, system_prompt=None):
        self.system_prompt = system_prompt
        return {
            "status": "success",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
            "llm_response": {"content": "Answer."},
        }


@pytest.mark.asyncio
async def test_success_unifies_top_k_and_returns_display_context() -> None:
    rag = SuccessfulFakeRAG()

    result = await LightRAGAdapter().query(
        rag,
        "Who is Alice?",
        QueryMode.MIX,
        5,
    )

    assert result.answer == "Alice is the protagonist."
    assert result.entities == [{"entity_name": "ALICE"}]
    assert "## Entities" in result.context_text
    assert "## Chunks" in result.context_text
    assert rag.param.mode == "mix"
    assert rag.param.top_k == 5
    assert rag.param.chunk_top_k == 5
    assert rag.param.stream is False


@pytest.mark.asyncio
async def test_raised_timeout_is_mapped_without_private_details() -> None:
    with pytest.raises(AppError) as error:
        await LightRAGAdapter().query(
            RaisingFakeRAG(),
            "Who is Alice?",
            QueryMode.MIX,
            5,
        )

    assert error.value.code == "LLM_TIMEOUT"
    assert "private details" not in str(error.value)


@pytest.mark.asyncio
async def test_failure_dictionary_is_rejected() -> None:
    with pytest.raises(LightRAGQueryError, match="no results"):
        await LightRAGAdapter().query(
            FailureFakeRAG(),
            "Who is Alice?",
            QueryMode.MIX,
            5,
        )


@pytest.mark.asyncio
async def test_missing_structured_lists_become_empty_with_warnings() -> None:
    result = await LightRAGAdapter().query(
        MissingDataFakeRAG(),
        "Who is Alice?",
        QueryMode.LOCAL,
        3,
    )

    assert result.entities == []
    assert result.relationships == []
    assert result.references == []
    assert len(result.warnings) == 3
    assert result.chunks == [{"content": "Only a chunk."}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "top_k"),
    [("   ", 5), ("question", 0), ("question", 51)],
)
async def test_invalid_query_parameters_are_rejected(query: str, top_k: int) -> None:
    with pytest.raises(AppError) as error:
        await LightRAGAdapter().query(
            SuccessfulFakeRAG(),
            query,
            QueryMode.HYBRID,
            top_k,
        )

    assert error.value.code == "INVALID_QUERY"


@pytest.mark.asyncio
async def test_optional_system_prompt_is_forwarded() -> None:
    rag = SystemPromptFakeRAG()

    await LightRAGAdapter().query(
        rag,
        "Who is Alice?",
        QueryMode.MIX,
        5,
        system_prompt="Use {context_data}",
    )

    assert rag.system_prompt == "Use {context_data}"
