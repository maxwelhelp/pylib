"""
Lifting API Server

Закрытый сервер с математическим ядром библиотеки.
Клиенты взаимодействуют через HTTP API.
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from .auth import get_api_key, check_rate_limit, APIKey
from .routes import geometry, sync, batch
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events"""
    print(f"Lifting API Server v{settings.VERSION} starting...")
    print(f"Rate limit: {settings.RATE_LIMIT_PER_MINUTE} req/min")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Lifting API",
    description="Geometric data representation on unit sphere",
    version=settings.VERSION,
    lifespan=lifespan,
)

# CORS для веб-клиентов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production ограничить
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# Middleware для логирования и метрик
@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response


# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}


# Routes
app.include_router(geometry.router, prefix="/geometry", tags=["geometry"])
app.include_router(sync.router, prefix="/sync", tags=["sync"])
app.include_router(batch.router, tags=["batch"])


# Usage info
@app.get("/v1/usage")
async def get_usage(api_key: APIKey = Depends(get_api_key)):
    """Получить информацию об использовании API"""
    return {
        "api_key": api_key.key[:8] + "...",
        "requests_today": api_key.requests_today,
        "limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
        "limit_per_day": settings.RATE_LIMIT_PER_DAY,
    }
