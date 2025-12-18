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
    from ..core import centroid, d_geo
except ImportError:
    from core import centroid, d_geo


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
        
        distances = np.array([d_geo(s, self.centroid) for s in shapes])
        
        self.mean_distance = float(np.mean(distances))
        self.std_distance = float(np.std(distances))
        
        if self.threshold_percentile is not None:
            self.threshold = float(np.percentile(distances, self.threshold_percentile))
        else:
            self.threshold = self.mean_distance + self.threshold_sigma * self.std_distance
        
        self._is_fitted = True
        return self
    
    def predict(self, shapes: NDArray) -> List[AnomalyResult]:
        """Детекция аномалий."""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        shapes = np.asarray(shapes)
        if shapes.ndim == 1:
            shapes = shapes.reshape(1, -1)
        
        results = []
        for shape in shapes:
            d = d_geo(shape, self.centroid)
            results.append(AnomalyResult(
                distance=d,
                distance_deg=np.degrees(d),
                is_anomaly=d > self.threshold,
                threshold=self.threshold,
            ))
        
        return results
    
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
