"""
Central configuration for the Image Analyzer.

Set these values with environment variables or edit the defaults below.
Do not commit real passwords or API keys to source control.
"""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Browser interface login
    APP_USERNAME: str = os.getenv("APP_USERNAME", "admin")
    APP_PASSWORD: str = os.getenv("APP_PASSWORD", "change-this-password")

    # Provider selection: "ollama" or "openai"
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "ollama").lower()

    # Ollama
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = os.getenv(
        "OLLAMA_MODEL",
        "pdurlej/qwythos-9b-claude-mythos-5-1m:latest"
    )
    OLLAMA_USERNAME: str = os.getenv("OLLAMA_USERNAME", "")
    OLLAMA_PASSWORD: str = os.getenv("OLLAMA_PASSWORD", "")

    # OpenAI SDK
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Upload limits
    MAX_IMAGES: int = int(os.getenv("MAX_IMAGES", "10"))
    MAX_IMAGE_MB: int = int(os.getenv("MAX_IMAGE_MB", "15"))


settings = Settings()
