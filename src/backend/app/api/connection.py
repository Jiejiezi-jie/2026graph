from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, field_validator


class ConnectionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr | None = None
    base_url: str = Field(default="https://api.deepseek.com", max_length=2048)

    @field_validator("api_key")
    @classmethod
    def valid_key(cls, value):
        if value is None:
            return None  # Omitted key keeps the current secret; it is never returned.
        key = value.get_secret_value().strip()
        if not key or len(key) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise ValueError("API Key 格式无效")
        return SecretStr(key)

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value):
        value = value.strip()
        if any(c.isspace() or ord(c) < 32 for c in value):
            raise ValueError("Base URL 格式无效")
        TypeAdapter(AnyHttpUrl).validate_python(value)
        parts = urlsplit(value)
        if not parts.hostname or parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            raise ValueError("Base URL 不能包含凭据、查询参数或片段")
        return value.rstrip("/")


class ConnectionStatus(BaseModel):
    base_url: str
    api_key_configured: bool
    llm_model: str
    session_override: bool


router = APIRouter(prefix="/api/settings")


@router.get("/connection", response_model=ConnectionStatus)
async def get_connection(request: Request):
    return request.app.state.runtime.connection_status()


@router.post("/connection", response_model=ConnectionStatus)
async def update_connection(payload: ConnectionUpdate, request: Request):
    origin = request.headers.get("origin")
    allowed = {"http://localhost:5173", "http://127.0.0.1:5173", str(request.base_url).rstrip("/")}
    if origin and origin not in allowed:
        raise HTTPException(status_code=403, detail="不能从其他网站修改本地 API 设置。")
    return await request.app.state.runtime.update_connection(payload)
