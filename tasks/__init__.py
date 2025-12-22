"""
Tasks - Готовые задачи

Универсальные задачи, использующие геометрию:
- anomaly: детекция аномалий
- classifier: классификация
- base: базовый класс для моделей
- backend: переключение локальный/удалённый API
"""

from .base import BaseModel

from .anomaly import (
    AnomalyDetector,
    AnomalyResult,
    detect_anomalies,
)

from .classifier import (
    Classifier,
    ClassificationResult,
    classify,
)

from .backend import (
    get_backend,
    set_backend,
    use_remote,
    use_local,
    backend_context,
    LocalBackend,
    RemoteBackend,
)

__all__ = [
    # Base
    'BaseModel',

    # Anomaly
    'AnomalyDetector',
    'AnomalyResult',
    'detect_anomalies',

    # Classifier
    'Classifier',
    'ClassificationResult',
    'classify',

    # Backend
    'get_backend',
    'set_backend',
    'use_remote',
    'use_local',
    'backend_context',
    'LocalBackend',
    'RemoteBackend',
]
