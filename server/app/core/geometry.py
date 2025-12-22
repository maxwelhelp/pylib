"""
Core Geometry - СКРЫТАЯ МАТЕМАТИКА

Этот файл содержит ключевые алгоритмы библиотеки.
НЕ ПУБЛИКОВАТЬ.
"""

import numpy as np
from numpy.typing import NDArray


def normalize(x: NDArray) -> NDArray:
    """
    Нормализация вектора → shape на единичной сфере.

    Shape = точка на S^{n-1}, ||shape|| = 1
    """
    x = np.asarray(x).flatten().astype(np.float64)
    norm = np.linalg.norm(x)
    if norm < 1e-10:
        # Для нулевого вектора — случайное направление
        x = np.random.randn(len(x))
        norm = np.linalg.norm(x)
    return x / norm


def normalize_batch(data: NDArray) -> NDArray:
    """
    Batch нормализация.

    Args:
        data: [N, dim] массив векторов

    Returns:
        [N, dim] массив shapes
    """
    data = np.asarray(data).astype(np.float64)
    if data.ndim == 1:
        return normalize(data).reshape(1, -1)

    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return data / norms


def d_geo(x: NDArray, y: NDArray) -> float:
    """
    Геодезическое расстояние между двумя shapes.

    d(x, y) = arccos(⟨x, y⟩)

    Returns:
        Угол в радианах [0, π]
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    dot = np.clip(np.dot(x, y), -1.0, 1.0)
    return float(np.arccos(dot))


def d_geo_batch(shapes: NDArray, target: NDArray) -> NDArray:
    """
    Расстояния от множества shapes до одной точки.

    Args:
        shapes: [N, dim]
        target: [dim]

    Returns:
        [N] массив расстояний в радианах
    """
    shapes = np.asarray(shapes)
    target = np.asarray(target).flatten()
    dots = np.dot(shapes, target)
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


def distance_matrix(shapes: NDArray) -> NDArray:
    """
    Матрица попарных расстояний (векторизовано).

    Args:
        shapes: [N, dim]

    Returns:
        [N, N] матрица расстояний
    """
    shapes = np.asarray(shapes)
    G = np.dot(shapes, shapes.T)
    G = np.clip(G, -1.0, 1.0)
    return np.arccos(G)


def centroid(shapes: NDArray) -> NDArray:
    """
    Центроид (среднее на сфере).

    centroid = normalize(mean(shapes))
    """
    shapes = np.asarray(shapes)
    mean = np.mean(shapes, axis=0)
    return normalize(mean)


def d_eff(shapes: NDArray, threshold: float = 0.95) -> int:
    """
    Эффективная размерность через SVD.

    d_eff = min{d : Σᵢ₌₁ᵈ σᵢ² ≥ threshold × Σᵢ σᵢ²}
    """
    shapes = np.asarray(shapes)
    if shapes.ndim == 1:
        return 1

    # SVD
    _, s, _ = np.linalg.svd(shapes, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()

    if total < 1e-10:
        return 1

    cumsum = np.cumsum(s2) / total
    d = int(np.searchsorted(cumsum, threshold) + 1)
    return min(d, len(s))


def slerp(x: NDArray, y: NDArray, t: float) -> NDArray:
    """
    Spherical Linear Interpolation.

    slerp(x, y, t) = sin((1-t)θ)/sin(θ) · x + sin(tθ)/sin(θ) · y
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    dot = np.clip(np.dot(x, y), -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-10:
        return x.copy()

    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * x + (np.sin(t * theta) / sin_theta) * y
