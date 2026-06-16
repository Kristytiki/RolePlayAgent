"""Centralised env-loaded settings."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    chai_api_key: str
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    session_dir: str = "./.sessions"
    log_level: str = "INFO"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    rag_k: int = 4
    enable_rag: bool = True
    enable_gca: bool = True
    enable_sat_format: bool = True
    scene_min_score: float = 0.30
    summary_ratio: float = 0.3
    summary_preserve_recent: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
