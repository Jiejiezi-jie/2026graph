import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.domain.errors import AppError
from app.medical.retrievers import AdaptiveRetriever
from app.retrieval.contracts import RetrievalRequest, RetrievalResult
from app.services.query_service import QueryService


@pytest.mark.asyncio
async def test_routing_and_evidence_arrive_before_generation_and_are_saved_once():
    events = []
    predictor = SimpleNamespace(
        predict=Mock(return_value=["lightrag"]),
        predict_proba=Mock(return_value=[[.2, .7, .1]]),
        classes_=["vector", "lightrag", "pathrag"],
    )

    async def retrieve(request):
        assert [event[0] for event in events] == ["routing"]
        assert request.options == {"mode": "hybrid", "graph_top_k": "40"}
        return RetrievalResult(context_text="real evidence", chunks=[{"content": "real evidence"}])

    async def generate(query, evidence):
        assert [event[0] for event in events] == ["routing", "retrieval"]
        assert events[-1][1]["retrieval"]["chunks"] == evidence.chunks
        return "answer"

    registry = SimpleNamespace()
    adaptive = AdaptiveRetriever(registry, lambda: predictor)
    registry.get = lambda method: adaptive if method == "adaptive" else SimpleNamespace(retrieve=retrieve)
    runs = SimpleNamespace(save=Mock())
    service = QueryService(registry, SimpleNamespace(generate=generate), runs, asyncio.Lock())

    async def emit(kind, data):
        events.append((kind, data))

    result = await service.query(RetrievalRequest(query="test", method_id="adaptive"), on_event=emit)
    assert result["answer"] == "answer"
    assert events[0][1]["id"] == result["id"]
    assert events[0][1]["metadata"] == result["retrieval"]["metadata"]
    predictor.predict.assert_called_once()
    predictor.predict_proba.assert_called_once()
    runs.save.assert_called_once()
    assert not service.lock.locked()


@pytest.mark.asyncio
async def test_retrieval_failure_keeps_completed_routing():
    async def route(request):
        return request.model_copy(update={"method_id": "pathrag"}), {"selected_method": "pathrag"}

    async def retrieve_routed(request, metadata):
        raise AppError("RETRIEVAL_FAILED", "检索失败")

    service = QueryService(SimpleNamespace(get=lambda _: SimpleNamespace(route=route, retrieve_routed=retrieve_routed)),
                           None, SimpleNamespace(save=Mock()), asyncio.Lock())
    result = await service.query(RetrievalRequest(query="test", method_id="adaptive"))
    assert result["status"] == "failure"
    assert result["retrieval"]["metadata"]["selected_method"] == "pathrag"


@pytest.mark.asyncio
async def test_cancellation_releases_query_lock():
    entered = asyncio.Event()

    async def retrieve(request):
        entered.set()
        await asyncio.Event().wait()

    lock = asyncio.Lock()
    service = QueryService(SimpleNamespace(get=lambda _: SimpleNamespace(retrieve=retrieve)), None,
                           SimpleNamespace(save=Mock()), lock)
    task = asyncio.create_task(service.query(RetrievalRequest(query="test")))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not lock.locked()


@pytest.mark.asyncio
async def test_stream_delivers_routing_while_retrieval_is_still_waiting():
    import json
    from app.services.query_stream import query_stream

    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def query(request, *, on_event):
        try:
            await on_event("routing", {"id": "live", "metadata": {"selected_method": "pathrag"}})
            await release.wait()
            return {"id": "live", "answer": "finished"}
        finally:
            cancelled.set()

    stream = query_stream(SimpleNamespace(query=query), RetrievalRequest(query="test"))
    event = json.loads(await asyncio.wait_for(anext(stream), 1))
    assert event["type"] == "routing"
    assert not release.is_set() and not cancelled.is_set()
    await stream.aclose()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_stream_delivers_routing_while_retrieval_is_still_waiting():
    import json
    from app.services.query_stream import query_stream

    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def query(request, *, on_event):
        try:
            await on_event("routing", {"id": "live", "metadata": {"selected_method": "pathrag"}})
            await release.wait()
            return {"id": "live", "answer": "finished"}
        finally:
            cancelled.set()

    stream = query_stream(SimpleNamespace(query=query), RetrievalRequest(query="test"))
    event = json.loads(await asyncio.wait_for(anext(stream), 1))
    assert event["type"] == "routing"
    assert not release.is_set() and not cancelled.is_set()
    await stream.aclose()
    assert cancelled.is_set()
