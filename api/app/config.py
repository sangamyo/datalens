from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables (see .env.example)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Defaults point at localhost so the app can be imported without Docker (e.g. in tests).
    database_url: str = "postgresql+psycopg://episodehub:change-me@localhost:5432/episodehub"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "change-me-too"
    s3_bucket: str = "episodehub"


@lru_cache
def get_settings() -> Settings:
    return Settings()
