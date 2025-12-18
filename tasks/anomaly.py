"""
Anomaly Detection Task

Универсальная детекция аномалий через геометрию.
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import pickle
from pathlib import Path

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

    Режимы порога:
        - threshold_sigma: порог = mean + sigma * std (по умолчанию)
        - threshold_percentile: порог = percentile расстояний
        - contamination: ожидаемая доля аномалий (авто-percentile)

    Examples:
        >>> detector = AnomalyDetector(threshold_sigma=2.0)
        >>> detector.fit(normal_shapes)
        >>> results = detector.predict(test_shapes)

        >>> # Или с contamination (5% аномалий):
        >>> detector = AnomalyDetector(contamination=0.05)
    """

    def __init__(self,
                 threshold_sigma: float = 2.0,
                 threshold_percentile: Optional[float] = None,
                 contamination: Optional[float] = None):
        """
        Args:
            threshold_sigma: множитель стандартного отклонения (default)
            threshold_percentile: фиксированный персентиль (0-100)
            contamination: ожидаемая доля аномалий (0-1), авто-персентиль
        """
        self.threshold_sigma = threshold_sigma
        self.threshold_percentile = threshold_percentile
        self.contamination = contamination

        self.centroid = None
        self.threshold = None
        self.mean_distance = None
        self.std_distance = None
        self._distances_hist = None  # для диагностики
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
        self._distances_hist = distances  # сохраняем для диагностики

        self.mean_distance = float(np.mean(distances))
        self.std_distance = float(np.std(distances))

        # Выбор порога (приоритет: contamination > percentile > sigma)
        if self.contamination is not None:
            # contamination = доля аномалий, значит порог = (1-contamination) персентиль
            percentile = 100 * (1 - self.contamination)
            self.threshold = float(np.percentile(distances, percentile))
        elif self.threshold_percentile is not None:
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
            [N] массив значений [0, 1+], где 1 = на пороге, >1 = аномалия
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        shapes = np.asarray(shapes)
        if shapes.ndim == 1:
            shapes = shapes.reshape(1, -1)

        distances = d_geo_batch(shapes, self.centroid)
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
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        return {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1': float(f1),
        }

    def get_stats(self) -> Dict[str, float]:
        """Статистика обученной модели."""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted.")
        return {
            'threshold': self.threshold,
            'threshold_deg': np.degrees(self.threshold),
            'mean_distance': self.mean_distance,
            'mean_distance_deg': np.degrees(self.mean_distance),
            'std_distance': self.std_distance,
            'std_distance_deg': np.degrees(self.std_distance),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Сохранить модель в файл."""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Nothing to save.")

        data = {
            'centroid': self.centroid,
            'threshold': self.threshold,
            'mean_distance': self.mean_distance,
            'std_distance': self.std_distance,
            'threshold_sigma': self.threshold_sigma,
            'threshold_percentile': self.threshold_percentile,
            'contamination': self.contamination,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AnomalyDetector':
        """Загрузить модель из файла."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        detector = cls(
            threshold_sigma=data.get('threshold_sigma', 2.0),
            threshold_percentile=data.get('threshold_percentile'),
            contamination=data.get('contamination'),
        )
        detector.centroid = data['centroid']
        detector.threshold = data['threshold']
        detector.mean_distance = data['mean_distance']
        detector.std_distance = data['std_distance']
        detector._is_fitted = True
        return detector


def detect_anomalies(shapes: NDArray,
                     threshold_sigma: float = 2.0,
                     labels: Optional[NDArray] = None) -> List[AnomalyResult]:
    """Быстрая детекция."""
    detector = AnomalyDetector(threshold_sigma=threshold_sigma)
    detector.fit(shapes, labels)
    return detector.predict(shapes)
