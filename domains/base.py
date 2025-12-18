"""
Base Domain Analyzer
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod

# Универсальные импорты
try:
    from ..core import normalize
    from ..tasks import AnomalyDetector, Classifier
except ImportError:
    from core import normalize
    from tasks import AnomalyDetector, Classifier


class BaseDomainAnalyzer(ABC):
    """
    Базовый класс для domain-specific анализа.
    
    Наследуй и реализуй:
    - preprocess(): raw → segments
    - extract_features(): segment → features dict
    """
    
    default_config: Dict[str, Any] = {}
    
    def __init__(self, **config):
        self.config = {**self.default_config, **config}
        self._detector = None
        self._classifier = None
        self._is_fitted = False
    
    @abstractmethod
    def preprocess(self, data: Any) -> List[Any]:
        """Сырые данные → список сегментов"""
        pass
    
    @abstractmethod
    def extract_features(self, segment: Any) -> Dict[str, float]:
        """Сегмент → словарь фичей"""
        pass
    
    def to_shape(self, features: Union[Dict, NDArray]) -> NDArray:
        """Features → shape на сфере"""
        if isinstance(features, dict):
            arr = np.array(list(features.values()), dtype=np.float64)
        else:
            arr = np.asarray(features, dtype=np.float64).flatten()
        return normalize(arr)
    
    def process(self, data: Any) -> NDArray:
        """Полный pipeline: data → shapes"""
        segments = self.preprocess(data)
        shapes = [self.to_shape(self.extract_features(seg)) for seg in segments]
        return np.array(shapes)
    
    def fit_anomaly(self, data: Any, labels: Optional[List] = None,
                    threshold_sigma: float = 2.0) -> 'BaseDomainAnalyzer':
        """Обучение детектора аномалий"""
        shapes = self.process(data)
        self._detector = AnomalyDetector(threshold_sigma=threshold_sigma)
        self._detector.fit(shapes, labels)
        self._is_fitted = True
        return self
    
    def detect_anomalies(self, data: Any) -> List[Dict]:
        """Детекция аномалий"""
        if self._detector is None:
            raise RuntimeError("Call fit_anomaly() first")
        shapes = self.process(data)
        results = self._detector.predict(shapes)
        return [{'distance': r.distance, 'distance_deg': r.distance_deg,
                 'is_anomaly': r.is_anomaly} for r in results]
    
    def fit_classifier(self, data: Any, labels: List) -> 'BaseDomainAnalyzer':
        """Обучение классификатора"""
        shapes = self.process(data)
        self._classifier = Classifier()
        self._classifier.fit(shapes, labels)
        self._is_fitted = True
        return self
    
    def classify(self, data: Any) -> List[Dict]:
        """Классификация"""
        if self._classifier is None:
            raise RuntimeError("Call fit_classifier() first")
        shapes = self.process(data)
        results = self._classifier.predict(shapes)
        return [{'predicted': r.predicted, 'confidence': r.confidence} for r in results]
