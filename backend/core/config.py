"""
Configuration management using Pydantic Settings

Loads from environment variables with sensible defaults
"""
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_PREFIX: str = "/api"
    DEBUG: bool = False

    # Database (relative to project root)
    DATABASE_PATH: str = "../data/analytics.duckdb"

    # Ollama/LLM
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_TIMEOUT: int = 30

    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 100
    UPLOAD_DIR: str = "../data/raw"

    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = "../.env"  # Look for .env in project root
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()

