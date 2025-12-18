"""
Tasks - Готовые задачи

Универсальные задачи, использующие геометрию:
- anomaly: детекция аномалий
- classifier: классификация
- base: базовый класс для моделей
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
]
