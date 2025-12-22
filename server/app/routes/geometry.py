"""
Geometry Routes - операции на сфере
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from ..auth import get_api_key, check_rate_limit, APIKey
from ..core import geometry as geo

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class NormalizeRequest(BaseModel):
    """Запрос на нормализацию"""
    data: List[List[float]] = Field(..., description="Массив векторов для нормализации")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [[1, 2, 3], [4, 5, 6]]
            }
        }


class NormalizeResponse(BaseModel):
    shapes: List[List[float]]
    count: int


class DistanceRequest(BaseModel):
    """Запрос на расчёт расстояний"""
    shapes: List[List[float]] = Field(..., description="Shapes для расчёта")
    target: Optional[List[float]] = Field(None, description="Целевая точка (если None — матрица расстояний)")


class DistanceResponse(BaseModel):
    distances: List[float] | List[List[float]]
    unit: str = "radians"


class CentroidRequest(BaseModel):
    """Запрос на расчёт центроида"""
    shapes: List[List[float]]


class CentroidResponse(BaseModel):
    centroid: List[float]
    count: int


# ============================================================
# Endpoints
# ============================================================

@router.post("/normalize", response_model=NormalizeResponse)
async def normalize(
    request: NormalizeRequest,
    api_key: APIKey = Depends(get_api_key),
):
    """
    Нормализовать векторы → shapes на единичной сфере.

    Shape = вектор с ||shape|| = 1
    """
    await check_rate_limit(api_key)

    data = np.array(request.data)
    shapes = geo.normalize_batch(data)

    return NormalizeResponse(
        shapes=shapes.tolist(),
        count=len(shapes),
    )


@router.post("/distance", response_model=DistanceResponse)
async def distance(
    request: DistanceRequest,
    api_key: APIKey = Depends(get_api_key),
):
    """
    Расчёт геодезических расстояний.

    Если target указан — расстояния от каждого shape до target.
    Если target не указан — матрица попарных расстояний.
    """
    await check_rate_limit(api_key)

    shapes = np.array(request.shapes)

    if request.target is not None:
        target = np.array(request.target)
        distances = geo.d_geo_batch(shapes, target)
        return DistanceResponse(distances=distances.tolist())
    else:
        distances = geo.distance_matrix(shapes)
        return DistanceResponse(distances=distances.tolist())


@router.post("/centroid", response_model=CentroidResponse)
async def centroid(
    request: CentroidRequest,
    api_key: APIKey = Depends(get_api_key),
):
    """
    Расчёт центроида (среднее на сфере).
    """
    await check_rate_limit(api_key)

    shapes = np.array(request.shapes)
    c = geo.centroid(shapes)

    return CentroidResponse(
        centroid=c.tolist(),
        count=len(shapes),
    )
