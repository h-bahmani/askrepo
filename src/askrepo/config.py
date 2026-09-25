"""Environment-driven settings. Copy .env.example to .env and fill in your values."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASKREPO_", env_file=".env", extra="ignore")

    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None

    index_path: str = ".askrepo/index.pkl"
    max_chunk_lines: int = 60
    chunk_overlap: int = 10


def get_settings() -> Settings:
    return Settings()
