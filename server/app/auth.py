"""
Authentication and Rate Limiting
"""

from fastapi import HTTPException, Header, Request
from fastapi.security import APIKeyHeader
from typing import Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import time
import asyncio

from .config import settings


@dataclass
class APIKey:
    """API Key информация"""
    key: str
    name: str
    is_premium: bool = False
    requests_today: int = 0
    requests_this_minute: int = 0
    last_request_time: float = 0
    minute_start: float = 0


# In-memory хранилище (в production — Redis)
_api_keys: Dict[str, APIKey] = {}
_rate_limits: Dict[str, list] = defaultdict(list)  # key -> [timestamps]
_lock = asyncio.Lock()


def _init_api_keys():
    """Инициализация API keys из конфига"""
    if _api_keys:
        return

    for pair in settings.API_KEYS.split(","):
        if ":" in pair:
            key, name = pair.strip().split(":", 1)
            _api_keys[key] = APIKey(key=key, name=name)


_init_api_keys()


async def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> APIKey:
    """
    Извлечь и проверить API key.

    Поддерживает:
    - Header: X-API-Key: your-key
    - Header: Authorization: Bearer your-key
    """
    api_key = None

    if x_api_key:
        api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Use X-API-Key header or Authorization: Bearer <key>",
        )

    if api_key not in _api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return _api_keys[api_key]


async def check_rate_limit(api_key: APIKey) -> None:
    """
    Проверить rate limit.

    Sliding window algorithm.
    """
    async with _lock:
        now = time.time()
        minute_ago = now - 60
        day_ago = now - 86400

        # Получаем timestamps запросов
        timestamps = _rate_limits[api_key.key]

        # Удаляем старые (больше дня)
        timestamps = [t for t in timestamps if t > day_ago]
        _rate_limits[api_key.key] = timestamps

        # Считаем запросы за минуту и день
        requests_this_minute = sum(1 for t in timestamps if t > minute_ago)
        requests_today = len(timestamps)

        # Лимиты
        limit_per_minute = (
            settings.PREMIUM_RATE_LIMIT_PER_MINUTE
            if api_key.is_premium
            else settings.RATE_LIMIT_PER_MINUTE
        )
        limit_per_day = (
            settings.PREMIUM_RATE_LIMIT_PER_DAY
            if api_key.is_premium
            else settings.RATE_LIMIT_PER_DAY
        )

        if requests_this_minute >= limit_per_minute:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": limit_per_minute,
                    "period": "minute",
                    "retry_after": 60 - (now - min(t for t in timestamps if t > minute_ago)),
                },
            )

        if requests_today >= limit_per_day:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Daily limit exceeded",
                    "limit": limit_per_day,
                    "period": "day",
                },
            )

        # Записываем запрос
        timestamps.append(now)
        api_key.requests_today = requests_today + 1
        api_key.requests_this_minute = requests_this_minute + 1


def create_api_key(name: str, is_premium: bool = False) -> str:
    """Создать новый API key (для админки)"""
    import secrets

    key = f"lft_{secrets.token_urlsafe(24)}"
    _api_keys[key] = APIKey(key=key, name=name, is_premium=is_premium)
    return key
