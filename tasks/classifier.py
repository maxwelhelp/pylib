"""
Classifier Task

Классификация через геометрию.
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

from .base import BaseModel
from .backend import get_backend


@dataclass
class ClassificationResult:
    """Результат классификации"""
    predicted: str
    distances: Dict[str, float]
    confidence: float

    def __repr__(self):
        return f"ClassificationResult('{self.predicted}', conf={self.confidence:.2f})"


class Classifier(BaseModel):
    """
    Классификатор на основе расстояния до центроидов.

    Examples:
        >>> classifier = Classifier()
        >>> classifier.fit(train_shapes, train_labels)
        >>> predictions = classifier.predict(test_shapes)

        >>> # Сохранение (безопасный JSON):
        >>> classifier.save('model.json')
        >>> classifier = Classifier.load('model.json')
    """

    def __init__(self):
        super().__init__()
        self.centroids: Dict[str, NDArray] = {}
        self.classes: List[str] = []

    def _get_state(self) -> Dict[str, Any]:
        """Состояние для сериализации."""
        return {
            'centroids': self.centroids,
            'classes': self.classes,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Восстановление из сериализации."""
        self.centroids = state['centroids']
        self.classes = state['classes']
    
    def fit(self, shapes: NDArray, labels: Union[NDArray, List]) -> 'Classifier':
        """Обучение: центроид каждого класса."""
        shapes = np.asarray(shapes)
        labels = np.asarray(labels)

        self.classes = sorted(set(str(l) for l in labels))
        self.centroids = {}

        backend = get_backend()
        for cls in self.classes:
            idx = np.array([str(l) == cls for l in labels])
            class_shapes = shapes[idx]
            if len(class_shapes) > 0:
                self.centroids[cls] = backend.centroid(class_shapes)
        
        self._is_fitted = True
        return self
    
    def predict(self, shapes: NDArray) -> List[ClassificationResult]:
        """Классификация по ближайшему центроиду (векторизовано)."""
        self._check_fitted()

        shapes = np.asarray(shapes)
        if shapes.ndim == 1:
            shapes = shapes.reshape(1, -1)

        # Предвычисляем расстояния до всех центроидов батчем
        # distance_matrix[i, j] = расстояние от shapes[i] до centroids[j]
        centroid_array = np.array([self.centroids[cls] for cls in self.classes])
        # [N, num_classes]
        all_distances = np.dot(shapes, centroid_array.T)
        all_distances = np.clip(all_distances, -1.0, 1.0)
        all_distances = np.arccos(all_distances)

        results = []
        for i, shape in enumerate(shapes):
            distances = {cls: float(all_distances[i, j]) for j, cls in enumerate(self.classes)}
            predicted = min(distances, key=distances.get)

            sorted_dists = sorted(distances.values())
            confidence = sorted_dists[1] - sorted_dists[0] if len(sorted_dists) >= 2 else float('inf')

            results.append(ClassificationResult(
                predicted=predicted,
                distances=distances,
                confidence=float(confidence),
            ))

        return results

    def predict_labels(self, shapes: NDArray) -> List[str]:
        """Быстрое предсказание только меток (без confidence)."""
        self._check_fitted()

        shapes = np.asarray(shapes)
        if shapes.ndim == 1:
            shapes = shapes.reshape(1, -1)

        centroid_array = np.array([self.centroids[cls] for cls in self.classes])
        all_distances = np.dot(shapes, centroid_array.T)
        all_distances = np.clip(all_distances, -1.0, 1.0)
        all_distances = np.arccos(all_distances)

        # Индексы минимальных расстояний
        min_indices = np.argmin(all_distances, axis=1)
        return [self.classes[i] for i in min_indices]
    
    def score(self, shapes: NDArray, labels: Union[NDArray, List]) -> float:
        """Accuracy."""
        results = self.predict(shapes)
        predictions = [r.predicted for r in results]
        labels = [str(l) for l in labels]

        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(labels) if len(labels) > 0 else 0.0


def classify(train_shapes: NDArray, train_labels: Union[NDArray, List],
             test_shapes: NDArray) -> List[ClassificationResult]:
    """Быстрая классификация."""
    classifier = Classifier()
    classifier.fit(train_shapes, train_labels)
    return classifier.predict(test_shapes)
