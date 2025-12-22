"""
Server Configuration
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Настройки сервера"""

    VERSION: str = "1.0.0"

    # API Keys (в production - из базы данных)
    # Формат: key1:name1,key2:name2
    API_KEYS: str = "demo-key:demo,dev-key-12345:developer,test-key-67890:tester"

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100  # Бесплатный лимит
    RATE_LIMIT_PER_DAY: int = 10000

    # Premium limits (для будущего)
    PREMIUM_RATE_LIMIT_PER_MINUTE: int = 1000
    PREMIUM_RATE_LIMIT_PER_DAY: int = 100000

    # Redis для rate limiting (опционально)
    REDIS_URL: Optional[str] = None

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
