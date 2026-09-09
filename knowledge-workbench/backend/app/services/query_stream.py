import asyncio
import json
from contextlib import suppress

from app.services.web_errors import safe_error


async def query_stream(service, request):
    """A single query publishes evidence before answer generation finishes."""
    queue = asyncio.Queue()

    async def emit(kind, data):
        await queue.put({"type": kind, **data})

    async def run():
        try:
            result = await service.query(request, on_event=emit)
            await emit("result", {"run": result})
        except Exception as exc:
            error = safe_error(exc)
            await emit("error", {"error": {"code": error.code, "message": error.safe_message}})

    task = asyncio.create_task(run())
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                event = {"type": "heartbeat"}
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if event["type"] in {"result", "error"}:
                break
    finally:
        # A disconnected browser must not leave a hidden query holding the workspace gate.
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
