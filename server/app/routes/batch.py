"""
Batch Routes - множественные операции за один запрос

Ключевой endpoint для минимизации latency.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import numpy as np

from ..auth import get_api_key, check_rate_limit, APIKey
from ..core import geometry as geo
from ..core import sync_lifting

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class BatchOperation(BaseModel):
    """Одна операция в batch"""
    op: str = Field(..., description="Тип операции: normalize, distance, centroid, sync, features")
    id: Optional[str] = Field(None, description="ID операции для ссылок")

    # Данные (зависят от операции)
    data: Optional[List[List[float]]] = None  # для normalize
    shapes: Optional[Union[List[List[float]], str]] = None  # для distance/centroid (или ссылка)
    target: Optional[Union[List[float], str]] = None  # для distance
    signal: Optional[List[float]] = None  # для sync
    signals: Optional[List[List[float]]] = None  # для batch sync
    transform: Optional[str] = "fft"
    transforms: Optional[List[str]] = None
    moments: Optional[List[int]] = [0, 1, 2]


class BatchRequest(BaseModel):
    """Batch запрос"""
    operations: List[BatchOperation] = Field(..., description="Список операций")

    class Config:
        json_schema_extra = {
            "example": {
                "operations": [
                    {"op": "normalize", "id": "shapes", "data": [[1, 2, 3], [4, 5, 6]]},
                    {"op": "centroid", "shapes": "ref:shapes"},
                    {"op": "distance", "shapes": "ref:shapes", "target": "ref:centroid"},
                ]
            }
        }


class BatchResult(BaseModel):
    """Результат одной операции"""
    id: Optional[str] = None
    op: str
    result: Any
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """Ответ batch"""
    results: List[BatchResult]
    operations_count: int
    success_count: int


# ============================================================
# Batch Processor
# ============================================================

async def process_batch(operations: List[BatchOperation]) -> List[BatchResult]:
    """Обработать batch операций"""
    results = []
    context = {}  # для хранения промежуточных результатов

    for op in operations:
        try:
            result = await process_operation(op, context)
            results.append(BatchResult(
                id=op.id,
                op=op.op,
                result=result,
            ))
            # Сохраняем результат для ссылок
            if op.id:
                context[op.id] = result
        except Exception as e:
            results.append(BatchResult(
                id=op.id,
                op=op.op,
                result=None,
                error=str(e),
            ))

    return results


def resolve_ref(value: Any, context: Dict) -> Any:
    """Разрешить ссылку на предыдущий результат"""
    if isinstance(value, str) and value.startswith("ref:"):
        ref_id = value[4:]
        if ref_id not in context:
            raise ValueError(f"Reference not found: {ref_id}")
        result = context[ref_id]
        # Извлекаем нужное поле
        if isinstance(result, dict):
            if "shapes" in result:
                return result["shapes"]
            if "centroid" in result:
                return result["centroid"]
            if "features" in result:
                return result["features"]
        return result
    return value


async def process_operation(op: BatchOperation, context: Dict) -> Dict:
    """Обработать одну операцию"""

    if op.op == "normalize":
        if op.data is None:
            raise ValueError("data required for normalize")
        data = np.array(op.data)
        shapes = geo.normalize_batch(data)
        return {"shapes": shapes.tolist(), "count": len(shapes)}

    elif op.op == "distance":
        shapes = resolve_ref(op.shapes, context)
        if shapes is None:
            raise ValueError("shapes required for distance")
        shapes = np.array(shapes)

        if op.target is not None:
            target = resolve_ref(op.target, context)
            target = np.array(target)
            distances = geo.d_geo_batch(shapes, target)
            return {"distances": distances.tolist()}
        else:
            distances = geo.distance_matrix(shapes)
            return {"distances": distances.tolist()}

    elif op.op == "centroid":
        shapes = resolve_ref(op.shapes, context)
        if shapes is None:
            raise ValueError("shapes required for centroid")
        shapes = np.array(shapes)
        c = geo.centroid(shapes)
        return {"centroid": c.tolist()}

    elif op.op == "sync":
        if op.signal is None:
            raise ValueError("signal required for sync")
        signal = np.array(op.signal)
        result = sync_lifting.compute(
            signal,
            transform=op.transform or "fft",
            moments=op.moments or [0, 1, 2],
        )
        return {
            "magnitude": result["magnitude"].tolist(),
            "time_center": result.get("time_center", np.array([])).tolist(),
            "time_spread": result.get("time_spread", np.array([])).tolist(),
        }

    elif op.op == "features":
        if op.signal is None:
            raise ValueError("signal required for features")
        signal = np.array(op.signal)
        features = sync_lifting.get_unified_features(
            signal,
            transforms=op.transforms or ["fft", "dct"],
            moments=op.moments or [0, 1, 2],
        )
        return {"features": features}

    elif op.op == "batch_features":
        # Batch обработка нескольких сигналов
        if op.signals is None:
            raise ValueError("signals required for batch_features")
        results = []
        for sig in op.signals:
            signal = np.array(sig)
            features = sync_lifting.get_unified_features(
                signal,
                transforms=op.transforms or ["fft", "dct"],
                moments=op.moments or [0, 1, 2],
            )
            results.append(features)
        return {"features_batch": results}

    else:
        raise ValueError(f"Unknown operation: {op.op}")


# ============================================================
# Endpoint
# ============================================================

@router.post("/batch", response_model=BatchResponse)
async def batch(
    request: BatchRequest,
    api_key: APIKey = Depends(get_api_key),
):
    """
    Выполнить множество операций за один запрос.

    Поддерживает ссылки между операциями через "ref:id".

    Пример:
    ```json
    {
        "operations": [
            {"op": "normalize", "id": "shapes", "data": [[1,2,3], [4,5,6]]},
            {"op": "centroid", "id": "center", "shapes": "ref:shapes"},
            {"op": "distance", "shapes": "ref:shapes", "target": "ref:center"}
        ]
    }
    ```
    """
    await check_rate_limit(api_key)

    if len(request.operations) > 100:
        raise HTTPException(
            status_code=400,
            detail="Too many operations (max 100)",
        )

    results = await process_batch(request.operations)
    success_count = sum(1 for r in results if r.error is None)

    return BatchResponse(
        results=results,
        operations_count=len(results),
        success_count=success_count,
    )
