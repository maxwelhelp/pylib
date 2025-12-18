"""
Classifier Task

Классификация через геометрию.
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

# Универсальные импорты
try:
    from ..core import centroid, d_geo
except ImportError:
    from core import centroid, d_geo


@dataclass
class ClassificationResult:
    """Результат классификации"""
    predicted: str
    distances: Dict[str, float]
    confidence: float
    
    def __repr__(self):
        return f"ClassificationResult('{self.predicted}', conf={self.confidence:.2f})"


class Classifier:
    """
    Классификатор на основе расстояния до центроидов.
    
    Examples:
        >>> classifier = Classifier()
        >>> classifier.fit(train_shapes, train_labels)
        >>> predictions = classifier.predict(test_shapes)
    """
    
    def __init__(self):
        self.centroids: Dict[str, NDArray] = {}
        self.classes: List[str] = []
        self._is_fitted = False
    
    def fit(self, shapes: NDArray, labels: Union[NDArray, List]) -> 'Classifier':
        """Обучение: центроид каждого класса."""
        shapes = np.asarray(shapes)
        labels = np.asarray(labels)
        
        self.classes = sorted(set(str(l) for l in labels))
        self.centroids = {}
        
        for cls in self.classes:
            idx = np.array([str(l) == cls for l in labels])
            class_shapes = shapes[idx]
            if len(class_shapes) > 0:
                self.centroids[cls] = centroid(class_shapes)
        
        self._is_fitted = True
        return self
    
    def predict(self, shapes: NDArray) -> List[ClassificationResult]:
        """Классификация по ближайшему центроиду."""
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        shapes = np.asarray(shapes)
        if shapes.ndim == 1:
            shapes = shapes.reshape(1, -1)
        
        results = []
        for shape in shapes:
            distances = {cls: d_geo(shape, c) for cls, c in self.centroids.items()}
            predicted = min(distances, key=distances.get)
            
            sorted_dists = sorted(distances.values())
            confidence = sorted_dists[1] - sorted_dists[0] if len(sorted_dists) >= 2 else float('inf')
            
            results.append(ClassificationResult(
                predicted=predicted,
                distances={k: float(v) for k, v in distances.items()},
                confidence=float(confidence),
            ))
        
        return results
    
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
