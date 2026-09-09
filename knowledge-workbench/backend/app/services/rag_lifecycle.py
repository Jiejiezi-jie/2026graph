"""Cleanup for the specifically pinned LightRAG runtime, not a version shim."""
import asyncio


async def finalize_rag(rag) -> None:
    failure = None
    try:
        await rag.finalize_storages()
    except BaseException as exc:
        failure = exc
    # The pinned finalize_storages closes stores, not the per-instance queues.
    # Retire them before a subsequent rebuild resets global namespace caches.
    states = getattr(rag, "_role_llm_states", {}) or {}
    functions = [getattr(state, "wrapped", None) for state in states.values()]
    functions.append(getattr(getattr(rag, "embedding_func", None), "func", None))
    functions.append(getattr(rag, "rerank_model_func", None))
    seen, shutdowns = set(), []
    for function in functions:
        shutdown = getattr(function, "shutdown", None)
        if shutdown is not None and id(function) not in seen:
            seen.add(id(function))
            shutdowns.append(shutdown(graceful=True, timeout=5))
    results = await asyncio.gather(*shutdowns, return_exceptions=True)
    if failure is not None:
        raise failure
    for result in results:
        if isinstance(result, BaseException):
            raise result
