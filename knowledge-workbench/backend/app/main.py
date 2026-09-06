from pathlib import Path
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel
from typing import Literal

from app.adapters.graphrag_bench import GraphRAGBenchLoader
from app.api.dependencies import AppContext
from app.api.connection import router as connection_router
from app.api.schemas import HealthResponse, DatasetStatusResponse, IndexStatusResponse
from app.config import Settings
from app.services.dataset_service import DatasetService
from app.services.lightrag_factory import LightRAGFactory
from app.services.workspace_service import WorkspaceService
from app.services.web_runtime import WebRuntime
from app.domain.errors import AppError
from app.retrieval.contracts import RetrievalRequest


APP_ROOT = Path(__file__).resolve().parents[2]


def create_default_context() -> AppContext:
    settings = Settings(app_root=APP_ROOT, _env_file=APP_ROOT / ".env")
    factory = LightRAGFactory(settings)
    return AppContext(
        settings=settings,
        dataset_service=DatasetService(GraphRAGBenchLoader(), settings.active_dir),
        workspace_service=WorkspaceService(
            settings.workspace_dir,
            factory,
            factory.runtime_fingerprint(),
        ),
    )


class DatasetSelection(BaseModel):
    subset: Literal["novel", "medical"] = "novel"


class IndexRequest(BaseModel):
    rebuild: bool = False


def create_app(context: AppContext | None = None, *, registry=None, generator=None) -> FastAPI:
    current_context = context or create_default_context()
    runtime = WebRuntime(current_context, registry, generator)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await runtime.close()

    app = FastAPI(title="LightRAG Retrieval Workbench", version="0.2.0", lifespan=lifespan)
    app.state.context = current_context
    app.state.runtime = runtime
    app.include_router(connection_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError):
        code = 400
        if exc.code in {"WORKSPACE_BUSY", "INDEX_NOT_READY", "REBUILD_CONFIRMATION_REQUIRED"}:
            code = 409
        elif exc.code in {"RUN_NOT_FOUND", "GRAPH_NOT_FOUND"}:
            code = 404
        return JSONResponse(status_code=code, content={"status": "failure", "error": {"code": exc.code, "message": exc.safe_message}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if request.url.path == "/api/settings/connection":
            # Pydantic's default errors may include raw invalid input, including keys.
            return JSONResponse(status_code=422, content={"status": "failure", "error": {
                "code": "INVALID_CONNECTION_SETTINGS",
                "message": "请检查 API Key 和 Base URL：Key 不能含空白字符，地址须为不含凭据或查询参数的 HTTP(S) 地址。",
            }})
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"status": "failure", "error": {"code": "INTERNAL_ERROR", "message": "后端处理失败，请检查本地文件与服务状态。"}})

    @app.get("/api/health", response_model=HealthResponse)
    def health(request: Request) -> dict:
        current: AppContext = request.app.state.context
        secret = current.settings.deepseek_api_key
        configured = bool(secret and secret.get_secret_value().strip())
        return {
            "status": "ok",
            "api_key_configured": configured,
            "llm_model": current.settings.llm_model,
            "embedding_model": current.settings.embedding_model,
            "graph_storage": current.settings.graph_storage,
        }

    @app.get("/api/dataset/status", response_model=DatasetStatusResponse)
    def dataset_status(request: Request) -> dict:
        current: AppContext = request.app.state.context
        return current.dataset_service.get_status()

    @app.get("/api/index/status", response_model=IndexStatusResponse)
    def index_status(request: Request) -> dict:
        return runtime.status()

    @app.post("/api/dataset/select", response_model=DatasetStatusResponse)
    async def select_dataset(selection: DatasetSelection):
        return await runtime.select(selection.subset)

    @app.post("/api/index/build")
    async def build_index(options: IndexRequest):
        return await runtime.start_build(options.rebuild)

    @app.get("/api/retrieval/methods")
    def methods():
        return {"methods": runtime.registry.descriptors()}

    @app.post("/api/query")
    async def query(payload: RetrievalRequest):
        return await runtime.queries.query(payload)

    @app.get("/api/graph")
    async def graph():
        if runtime.lock.locked():
            raise AppError("WORKSPACE_BUSY", "正在建索引或查询，请稍后读取图谱。")
        async with runtime.lock:
            if not runtime.status()["reusable"]:
                raise AppError("INDEX_NOT_READY", "当前数据尚无可复用的索引。")
            return await asyncio.to_thread(runtime.graph.preview)

    @app.get("/api/runs")
    def runs():
        return {"runs": runtime.runs.list_recent()}

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        result = runtime.runs.get(run_id)
        if result is None:
            raise AppError("RUN_NOT_FOUND", "找不到这条查询记录。")
        return result

    return app


app = create_app()
