"""
Tasks - Готовые задачи

Универсальные задачи, использующие геометрию:
- anomaly: детекция аномалий
- classifier: классификация
"""

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
    # Anomaly
    'AnomalyDetector',
    'AnomalyResult',
    'detect_anomalies',
    
    # Classifier
    'Classifier',
    'ClassificationResult',
    'classify',
]
