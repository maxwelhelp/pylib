"""
Sync Lifting Routes - синхронный лифтинг
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np

from ..auth import get_api_key, check_rate_limit, APIKey
from ..core import sync_lifting

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class SyncComputeRequest(BaseModel):
    """Запрос на SyncLifting.compute()"""
    signal: List[float] = Field(..., description="Входной сигнал")
    transform: str = Field("fft", description="Тип преобразования: fft, dct")
    moments: List[int] = Field([0, 1, 2], description="Моменты для вычисления")


class SyncComputeResponse(BaseModel):
    magnitude: List[float]
    time_center: Optional[List[float]] = None
    time_spread: Optional[List[float]] = None
    skewness: Optional[List[float]] = None


class SyncDominantRequest(BaseModel):
    """Запрос на доминирующие метрики"""
    signal: List[float]
    transform: str = "fft"
    moments: List[int] = [0, 1, 2]


class SyncDominantResponse(BaseModel):
    tc: float
    ts: float
    dominant_idx: int


class UnifiedFeaturesRequest(BaseModel):
    """Запрос на унифицированные фичи"""
    signal: List[float]
    transforms: List[str] = ["fft", "dct"]
    moments: List[int] = [0, 1, 2]


class UnifiedFeaturesResponse(BaseModel):
    features: Dict[str, float]


# ============================================================
# Endpoints
# ============================================================

@router.post("/compute", response_model=SyncComputeResponse)
async def sync_compute(
    request: SyncComputeRequest,
    api_key: APIKey = Depends(get_api_key),
):
    """
    Вычислить синхронные метрики для сигнала.

    Возвращает magnitude, time_center, time_spread для каждого коэффициента.
    """
    await check_rate_limit(api_key)

    signal = np.array(request.signal)
    result = sync_lifting.compute(
        signal,
        transform=request.transform,
        moments=request.moments,
    )

    response = SyncComputeResponse(
        magnitude=result["magnitude"].tolist(),
    )

    if "time_center" in result:
        response.time_center = result["time_center"].tolist()
    if "time_spread" in result:
        response.time_spread = result["time_spread"].tolist()
    if "skewness" in result:
        response.skewness = result["skewness"].tolist()

    return response


@router.post("/dominant", response_model=SyncDominantResponse)
async def sync_dominant(
    request: SyncDominantRequest,
    api_key: APIKey = Depends(get_api_key),
):
    """
    Получить метрики доминирующего коэффициента.

    Удобно когда нужна одна пара (tc, ts).
    """
    await check_rate_limit(api_key)

    signal = np.array(request.signal)
    result = sync_lifting.dominant_metrics(
        signal,
        transform=request.transform,
        moments=request.moments,
    )

    return SyncDominantResponse(
        tc=result["tc"],
        ts=result["ts"],
        dominant_idx=result["dominant_idx"],
    )


@router.post("/features", response_model=UnifiedFeaturesResponse)
async def unified_features(
    request: UnifiedFeaturesRequest,
    api_key: APIKey = Depends(get_api_key),
):
    """
    Унифицированный вектор фичей от нескольких преобразований.
    """
    await check_rate_limit(api_key)

    signal = np.array(request.signal)
    features = sync_lifting.get_unified_features(
        signal,
        transforms=request.transforms,
        moments=request.moments,
    )

    return UnifiedFeaturesResponse(features=features)
