"""
Configuration settings for the Multi-Tenant RAG System.

All values are sourced from environment variables (with .env fallback).
Critical fields are validated on startup to fail fast with clear messages.
"""
from functools import lru_cache
from typing import List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings driven by environment variables.

    Pydantic will raise a ``ValidationError`` at startup if required fields
    are missing or fail validation, preventing silent misconfigurations.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    app_name: str = Field(default="Multi-Tenant RAG System", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")

    # ── Databases ──────────────────────────────────────────────────────────
    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field(alias="REDIS_URL")

    # ── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")

    # ── Authentication ─────────────────────────────────────────────────────
    jwt_secret_key: str = Field(alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, alias="JWT_EXPIRE_MINUTES")

    # ── LLM Providers ─────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    default_llm_provider: str = Field(default="openai", alias="DEFAULT_LLM_PROVIDER")

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    # ── Security / CORS ────────────────────────────────────────────────────
    allowed_hosts: Union[List[str], str] = Field(
        default=["localhost", "127.0.0.1", "0.0.0.0"],
        alias="ALLOWED_HOSTS",
    )

    # ── File Upload ────────────────────────────────────────────────────────
    max_file_size_mb: int = Field(default=10, alias="MAX_FILE_SIZE_MB")
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    allowed_file_types: Union[List[str], str] = Field(
        default=["pdf", "txt", "docx"],
        alias="ALLOWED_FILE_TYPES",
    )

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Enforce a minimum secret length to prevent weak signing keys."""
        if len(v) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters long. "
                "Generate one with: openssl rand -hex 32"
            )
        return v

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v: Union[str, list]) -> List[str]:
        if isinstance(v, str):
            return [h.strip() for h in v.split(",") if h.strip()]
        return v

    @field_validator("allowed_file_types", mode="before")
    @classmethod
    def parse_allowed_file_types(cls, v: Union[str, list]) -> List[str]:
        if isinstance(v, str):
            return [t.strip().lower() for t in v.split(",") if t.strip()]
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Warn loudly if SQLite is used — it lacks PostgreSQL features."""
        if v.startswith("sqlite"):
            import warnings
            warnings.warn(
                "SQLite detected as database backend. "
                "This is only supported in testing. Use PostgreSQL in production.",
                stacklevel=2,
            )
        return v


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Module-level convenience alias
settings = get_settings()