"""
Core Shape - Нормализация и работа с shapes

Shape = нормализованный вектор на единичной сфере
S(x) = x / ||x||

После нормализации:
- Все shapes лежат на S^{n-1}
- ||S|| = 1
- Инвариантность к масштабу
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Optional
from dataclasses import dataclass


def normalize(x: NDArray, eps: float = 1e-10) -> NDArray:
    """
    Нормализация на единичную сферу
    
    S = x / ||x||
    
    Args:
        x: вектор или батч векторов
        eps: защита от деления на 0
        
    Returns:
        Нормализованный вектор с ||S|| = 1
        
    Examples:
        >>> s = normalize([3, 4])  # → [0.6, 0.8]
        >>> np.linalg.norm(s)      # → 1.0
    """
    x = np.asarray(x, dtype=np.float64)
    
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm < eps:
            # Случайное направление для нулевого вектора
            x = np.random.randn(*x.shape)
            norm = np.linalg.norm(x)
        return x / norm
    
    # Batch case
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def normalize_with_energy(x: NDArray, eps: float = 1e-10) -> Tuple[NDArray, float]:
    """
    Нормализация с сохранением энергии (нормы)
    
    Returns:
        (shape, energy) где energy = ||x||
        
    Полезно когда нужно восстановить исходный масштаб.
    """
    x = np.asarray(x, dtype=np.float64)
    energy = np.linalg.norm(x)
    
    if energy < eps:
        x = np.random.randn(*x.shape)
        energy = np.linalg.norm(x)
    
    return x / energy, energy


def denormalize(shape: NDArray, energy: float) -> NDArray:
    """Восстановление вектора из shape + energy"""
    return shape * energy


@dataclass
class Shape:
    """
    Контейнер для shape с метаданными
    
    Attributes:
        data: нормализованный вектор
        energy: исходная норма
        dim: размерность
    """
    data: NDArray
    energy: float = 1.0
    
    @property
    def dim(self) -> int:
        return len(self.data)
    
    @property
    def norm(self) -> float:
        return np.linalg.norm(self.data)
    
    def is_valid(self, tol: float = 1e-6) -> bool:
        """Проверка что ||data|| = 1"""
        return abs(self.norm - 1.0) < tol
    
    def to_numpy(self) -> NDArray:
        return self.data
    
    def __array__(self) -> NDArray:
        return self.data
    
    def __len__(self) -> int:
        return len(self.data)
    
    @classmethod
    def from_vector(cls, x: NDArray) -> 'Shape':
        """Создание Shape из вектора"""
        data, energy = normalize_with_energy(x)
        return cls(data=data, energy=energy)


def to_shape(x: NDArray) -> Shape:
    """
    Удобная функция для создания Shape
    
    Examples:
        >>> s = to_shape([3, 4, 0])
        >>> s.data   # → [0.6, 0.8, 0.0]
        >>> s.energy # → 5.0
    """
    return Shape.from_vector(x)


def dict_to_shape(features: dict) -> NDArray:
    """
    Словарь фичей → shape
    
    Args:
        features: {'fft_tc': 0.5, 'fft_ts': 0.2, ...}
        
    Returns:
        Нормализованный вектор
    """
    arr = np.array(list(features.values()), dtype=np.float64)
    return normalize(arr)


def concat_and_normalize(*arrays: NDArray) -> NDArray:
    """
    Объединение нескольких массивов и нормализация
    
    Полезно для комбинирования фичей из разных transforms.
    
    Examples:
        >>> s = concat_and_normalize(fft_features, wavelet_features)
    """
    concatenated = np.concatenate([np.asarray(a).flatten() for a in arrays])
    return normalize(concatenated)
