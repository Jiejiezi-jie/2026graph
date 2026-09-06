import asyncio
from datetime import datetime
from uuid import uuid4

from app.domain.errors import AppError
from app.retrieval.plugins import build_registry
from app.services.answer_service import DeepSeekAnswerGenerator
from app.services.graph_service import GraphService
from app.services.query_service import QueryService
from app.services.run_store import RunStore
from app.services.web_errors import safe_error
from app.services.workspace_service import GRAPH_FILE_NAME


class WebRuntime:
    """One process, one workspace. Index/query/select share the same mutation gate."""

    def __init__(self, context, registry=None, generator=None):
        self.context = context
        self.lock = asyncio.Lock()
        self.graph = GraphService(context.settings.workspace_dir / GRAPH_FILE_NAME)
        self.registry = registry if registry is not None else build_registry(context, self.graph)
        self.runs = RunStore(context.settings.app_root / "data" / "runs.sqlite3")
        self.queries = QueryService(self.registry, generator or DeepSeekAnswerGenerator(context.settings), self.runs, self.lock)
        self.build_task = None
        self.index_error = None
        self.session_override = False

    def connection_status(self):
        settings = self.context.settings
        key = settings.deepseek_api_key
        return {"base_url": settings.llm_base_url, "llm_model": settings.llm_model,
                "api_key_configured": bool(key and key.get_secret_value().strip()),
                "session_override": self.session_override}

    async def update_connection(self, payload):
        if self.lock.locked():
            raise AppError("WORKSPACE_BUSY", "正在建索引或查询，请完成后再修改 API 设置。")
        async with self.lock:
            settings = self.context.settings
            if payload.api_key is not None:
                settings.deepseek_api_key = payload.api_key
            settings.llm_base_url = payload.base_url
            workspace = self.context.workspace_service
            workspace.runtime_fingerprint = workspace.factory.runtime_fingerprint()
            self.session_override = True
            return self.connection_status()

    def status(self):
        try:
            status = self.context.workspace_service.get_status(self.context.dataset_service.load_active())
        except AppError as exc:
            if exc.code != "INVALID_INDEX_MANIFEST":
                raise
            status = {"status": "failed", "reusable": False, "corpus_name": None,
                      "started_at": None, "completed_at": None, "latency_ms": None,
                      "error_type": "INVALID_INDEX_MANIFEST"}
        if self.build_task is not None and not self.build_task.done():
            status["status"] = "building"
            status["reusable"] = False
        elif status["status"] == "building":
            status["status"] = "interrupted"
            status["reusable"] = False
        if self.index_error:
            status.update(status="failed", reusable=False, error_type=self.index_error["code"])
        return status

    async def select(self, subset: str):
        if self.lock.locked():
            raise AppError("WORKSPACE_BUSY", "正在建索引或查询，暂时不能更换数据。")
        async with self.lock:
            path = self.context.settings.benchmark_root / "Datasets" / "Corpus" / (subset + ".json")
            await asyncio.to_thread(self.context.dataset_service.select_shortest, path, subset)
            self.index_error = None
            return self.context.dataset_service.get_status()

    async def start_build(self, rebuild: bool):
        if self.lock.locked():
            raise AppError("WORKSPACE_BUSY", "正在建索引或查询，请勿重复启动。")
        document = self.context.dataset_service.load_active()
        if document is None:
            raise AppError("NO_DATASET", "请先选择一份文本。")
        status = self.status()
        if status["reusable"] and not rebuild:
            return {"action": "reused", "status": "ready"}
        secret = self.context.settings.deepseek_api_key
        if secret is None or not secret.get_secret_value().strip():
            raise AppError("MISSING_API_KEY", "尚未配置 API Key，请在网页的 API 设置中填写并保存。")
        workspace = self.context.settings.workspace_dir
        if workspace.exists() and any(workspace.iterdir()) and not rebuild:
            raise AppError("REBUILD_CONFIRMATION_REQUIRED", "已有索引不完整或不匹配，请确认备份后重建。")
        # Acquire before returning so another request cannot slip between task scheduling.
        await self.lock.acquire()
        self.index_error = None
        self.build_task = asyncio.create_task(self._build(document, rebuild))
        return {"action": "started", "status": "building"}

    async def _build(self, document, rebuild):
        try:
            if rebuild:
                self._archive_workspace()
            self.context.workspace_service.factory.reset_shared_storage()
            await self.context.workspace_service.ensure_ready(document)
        except Exception as exc:
            error = safe_error(exc)
            self.index_error = {"code": error.code, "message": error.safe_message}
        finally:
            self.lock.release()

    def _archive_workspace(self):
        data = (self.context.settings.app_root / "data").resolve()
        source = self.context.settings.workspace_dir
        if not source.exists():
            return
        # Reject junction/symlink escapes; move only the exact workspace.
        if source.resolve() != data / "workspace":
            raise AppError("UNSAFE_WORKSPACE_PATH", "Workspace 路径超出预期目录，未移动任何文件。")
        parent = data / "workspace_backups"
        if parent.exists() and parent.resolve() != parent:
            raise AppError("UNSAFE_BACKUP_PATH", "备份目录指向预期目录之外。")
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
        source.rename(target)

    async def close(self):
        if self.build_task is not None and not self.build_task.done():
            # Graceful shutdown waits for indexing. Force-kill will be shown as interrupted.
            await self.build_task
