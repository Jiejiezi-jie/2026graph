from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_BENCHMARK_ROOT = Path(
    r"D:\HuaweiMoveData\Users\huawei\Desktop\kg\GraphRAG-Benchmark"
)


class Settings(BaseSettings):
    """Non-UI runtime configuration for the V1 application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_root: Path = Field(default_factory=Path.cwd)
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "LLM_API_KEY"),
    )
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_max_token_size: int = 512
    chunk_token_size: int = 500
    chunk_overlap_token_size: int = 50
    graph_storage: str = "NetworkXStorage"

    @property
    def active_dir(self) -> Path:
        return self.app_root / "data" / "active"

    @property
    def workspace_dir(self) -> Path:
        return self.app_root / "data" / "workspace"

    @property
    def default_corpus_path(self) -> Path:
        return self.benchmark_root / "Datasets" / "Corpus" / "novel.json"
