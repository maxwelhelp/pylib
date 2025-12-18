"""
Anomaly Detection Task

Универсальная детекция аномалий через геометрию.
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Универсальные импорты
try:
    from ..core import centroid, d_geo, d_geo_batch
except ImportError:
    from core import centroid, d_geo, d_geo_batch


@dataclass
class AnomalyResult:
    """Результат детекции для одного sample"""
    distance: float
    distance_deg: float
    is_anomaly: bool
    threshold: float
    
    def __repr__(self):
        status = "ANOMALY" if self.is_anomaly else "normal"
        return f"AnomalyResult({self.distance_deg:.1f}°, {status})"


class AnomalyDetector:
    """
    Детектор аномалий на основе геометрического расстояния.
    
    Examples:
        >>> detector = AnomalyDetector(threshold_sigma=2.0)
        >>> detector.fit(normal_shapes)
        >>> results = detector.predict(test_shapes)
    """
    
    def __init__(self, 
                 threshold_sigma: float = 2.0,
                 threshold_percentile: Optional[float] = None):
        self.threshold_sigma = threshold_sigma
        self.threshold_percentile = threshold_percentile
        
        self.centroid = None
        self.threshold = None
        self.mean_distance = None
        self.std_distance = None
        self._is_fitted = False
    
    def fit(self, shapes: NDArray, labels: Optional[NDArray] = None) -> 'AnomalyDetector':
        """Обучение на нормальных данных."""
        shapes = np.asarray(shapes)
        
        if labels is not None:
            normal_idx = [i for i, l in enumerate(labels) if l in ['normal', 'N', 0, '0']]
            if len(normal_idx) > 0:
                shapes = shapes[normal_idx]
        
        self.centroid = centroid(shapes)

        # Векторизованный расчёт расстояний
        distances = d_geo_batch(shapes, self.centroid)
        
        self.mean_distance = float(np.mean(distances))
        self.std_distance = float(np.std(distances))
        
        if self.threshold_percentile is not None:
            self.threshold = float(np.percentile(distances, self.threshold_percentile))
        else:
            self.threshold = self.mean_distance + self.threshold_sigma * self.std_distance
        
        self._is_fitted = True
        return self
    
    def predict(self, shapes: NDArray) -> List[AnomalyResult]:
        """Детекция аномалий (векторизовано)."""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        shapes = np.asarray(shapes)
        if shapes.ndim == 1:
            shapes = shapes.reshape(1, -1)

        # Batch расчёт расстояний
        distances = d_geo_batch(shapes, self.centroid)
        is_anomaly = distances > self.threshold

        return [
            AnomalyResult(
                distance=float(d),
                distance_deg=float(np.degrees(d)),
                is_anomaly=bool(a),
                threshold=self.threshold,
            )
            for d, a in zip(distances, is_anomaly)
        ]

    def predict_proba(self, shapes: NDArray) -> NDArray:
        """
        Вероятность аномалии (нормализованное расстояние).

        Returns:
            [N] массив значений [0, 1], где 1 = точно аномалия
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        shapes = np.asarray(shapes)
        if shapes.ndim == 1:
            shapes = shapes.reshape(1, -1)

        distances = d_geo_batch(shapes, self.centroid)
        # Нормализуем: 0 = centroid, 1 = threshold, >1 = аномалия
        return distances / self.threshold
    
    def score(self, shapes: NDArray, labels: NDArray) -> Dict[str, float]:
        """Оценка качества."""
        results = self.predict(shapes)
        predictions = np.array([r.is_anomaly for r in results])
        labels = np.asarray(labels)
        
        is_anomaly = ~np.isin(labels, ['normal', 'N', 0, '0'])
        
        tp = np.sum(predictions & is_anomaly)
        tn = np.sum(~predictions & ~is_anomaly)
        fp = np.sum(predictions & ~is_anomaly)
        fn = np.sum(~predictions & is_anomaly)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
        }


def detect_anomalies(shapes: NDArray, 
                     threshold_sigma: float = 2.0,
                     labels: Optional[NDArray] = None) -> List[AnomalyResult]:
    """Быстрая детекция."""
    detector = AnomalyDetector(threshold_sigma=threshold_sigma)
    detector.fit(shapes, labels)
    return detector.predict(shapes)
