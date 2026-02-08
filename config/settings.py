"""Application configuration using Pydantic settings."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Top-level application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Core
    app_env: str = Field(default="development", alias="APP_ENV")
    app_name: str = Field(default="nemotron-ops-commander", alias="APP_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_key: str = Field(default="change-me", alias="API_KEY")

    # Nemotron
    nemotron_model_name: str = Field(
        default="nvidia/Nemotron-Mini-4B-Instruct", alias="NEMOTRON_MODEL_NAME"
    )
    nemotron_device: str = Field(default="cuda", alias="NEMOTRON_DEVICE")
    nemotron_max_new_tokens: int = Field(default=2048, alias="NEMOTRON_MAX_NEW_TOKENS")
    nemotron_temperature: float = Field(default=0.7, alias="NEMOTRON_TEMPERATURE")
    nemotron_top_p: float = Field(default=0.9, alias="NEMOTRON_TOP_P")
    nemotron_use_sglang: bool = Field(default=True, alias="NEMOTRON_USE_SGLANG")

    # ChromaDB (embedded, no server needed)
    chroma_persist_directory: str = Field(
        default="./chroma_storage", alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection: str = Field(default="incidents", alias="CHROMA_COLLECTION")

    # Observability
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    prometheus_port: int = Field(default=9000, alias="PROMETHEUS_PORT")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")


@lru_cache
def get_settings() -> AppSettings:
    """Return cached settings instance."""
    return AppSettings()
