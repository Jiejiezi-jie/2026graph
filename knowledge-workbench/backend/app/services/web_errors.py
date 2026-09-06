from app.domain.errors import AppError


def safe_error(exc: Exception) -> AppError:
    """Never expose raw provider exceptions, requests, or credentials to the UI."""
    if isinstance(exc, AppError):
        return exc
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return AppError("LLM_TIMEOUT", "模型请求超时，请稍后重试。")
    if "authentication" in name or "permission" in name:
        return AppError("LLM_AUTH_FAILED", "模型鉴权失败，请检查后端 API Key。")
    if "ratelimit" in name:
        return AppError("LLM_RATE_LIMITED", "模型服务限流，请稍后重试。")
    return AppError("QUERY_FAILED", "请求未完成，请检查模型连接及本地索引。")
