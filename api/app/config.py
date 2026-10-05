from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables (see .env.example)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Defaults point at localhost so the app can be imported without Docker (e.g. in tests).
    database_url: str = "postgresql+psycopg://datalens:change-me@localhost:5432/datalens"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint_url: str = "http://localhost:8333"
    s3_access_key: str = "datalens"
    s3_secret_key: str = "change-me-too"
    s3_bucket: str = "datalens"

    # Optional: Hugging Face token (public datasets work without it).
    hf_token: str | None = None
    # Optional LLM for natural-language search; rule-based parsing is used when neither is set.
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
