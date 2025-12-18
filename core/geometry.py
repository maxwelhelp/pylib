"""
Core Geometry - Универсальная математика на многообразиях

Это ЯДРО библиотеки. Не трогать без веской причины.

Содержит:
- Sphere: единичная сфера (основное пространство)
- Базовые операции: distance, centroid, slerp, frechet_mean
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List
from abc import ABC, abstractmethod


class Manifold(ABC):
    """Абстрактный базовый класс для многообразия"""
    
    @abstractmethod
    def distance(self, x: NDArray, y: NDArray) -> float:
        """Расстояние между точками"""
        pass
    
    @abstractmethod
    def centroid(self, points: NDArray) -> NDArray:
        """Центроид (среднее) множества точек"""
        pass


class Sphere(Manifold):
    """
    Единичная сфера S^{n-1} в R^n
    
    Основное пространство для shape-представлений.
    Все нормализованные вектора живут на сфере.
    
    Ключевые формулы:
        distance: d(x,y) = arccos(⟨x,y⟩)
        slerp:    S(t) = sin((1-t)θ)/sin(θ) · x + sin(tθ)/sin(θ) · y
        centroid: normalize(mean(points))
    """
    
    def distance(self, x: NDArray, y: NDArray) -> float:
        """
        Геодезическое расстояние (угол в радианах)
        
        d(x, y) = arccos(⟨x, y⟩)
        
        Returns:
            Угол в радианах [0, π]
        """
        x = self._ensure_normalized(x)
        y = self._ensure_normalized(y)
        dot = np.clip(np.dot(x.flatten(), y.flatten()), -1.0, 1.0)
        return np.arccos(dot)
    
    def centroid(self, points: NDArray) -> NDArray:
        """
        Центроид (упрощённое среднее Фреше)
        
        centroid = normalize(mean(points))
        
        Для точек близких друг к другу это хорошее приближение.
        """
        mean = np.mean(points, axis=0)
        return self._normalize(mean)
    
    def slerp(self, x: NDArray, y: NDArray, t: float) -> NDArray:
        """
        Spherical Linear Interpolation
        
        slerp(x, y, t) = sin((1-t)θ)/sin(θ) · x + sin(tθ)/sin(θ) · y
        
        Args:
            x, y: точки на сфере
            t: параметр [0, 1], где t=0 → x, t=1 → y
            
        Returns:
            Точка на геодезической между x и y
        """
        x = self._ensure_normalized(x)
        y = self._ensure_normalized(y)
        
        dot = np.clip(np.dot(x.flatten(), y.flatten()), -1.0, 1.0)
        theta = np.arccos(dot)
        
        if theta < 1e-10:
            return x.copy()
        
        sin_theta = np.sin(theta)
        return (np.sin((1-t) * theta) / sin_theta) * x + (np.sin(t * theta) / sin_theta) * y
    
    def frechet_mean(self, points: NDArray, 
                     max_iter: int = 100, 
                     tol: float = 1e-6) -> NDArray:
        """
        Среднее Фреше (точное геометрическое среднее)
        
        μ = argmin_m Σ d²(m, x_i)
        
        Итеративный алгоритм через exp/log maps.
        Для большинства случаев centroid() достаточно.
        """
        # Начальное приближение
        mean = self.centroid(points)
        
        for _ in range(max_iter):
            # Градиент в касательном пространстве
            tangent_sum = np.zeros_like(mean)
            for p in points:
                tangent_sum += self.log_map(mean, p)
            tangent_sum /= len(points)
            
            # Шаг
            mean_new = self.exp_map(mean, tangent_sum)
            
            if self.distance(mean, mean_new) < tol:
                break
            mean = mean_new
        
        return self._normalize(mean)
    
    def exp_map(self, base: NDArray, tangent: NDArray) -> NDArray:
        """
        Exponential map: касательное → сфера
        
        exp_x(v) = cos(||v||)·x + sin(||v||)·v/||v||
        """
        base = self._ensure_normalized(base)
        norm_v = np.linalg.norm(tangent)
        
        if norm_v < 1e-10:
            return base.copy()
        
        return np.cos(norm_v) * base + np.sin(norm_v) * tangent / norm_v
    
    def log_map(self, base: NDArray, point: NDArray) -> NDArray:
        """
        Logarithmic map: сфера → касательное
        
        log_x(y) = θ/sin(θ) · (y - cos(θ)·x)
        """
        base = self._ensure_normalized(base)
        point = self._ensure_normalized(point)
        
        dot = np.clip(np.dot(base.flatten(), point.flatten()), -1.0, 1.0)
        theta = np.arccos(dot)
        
        if theta < 1e-10:
            return np.zeros_like(base)
        
        direction = point - dot * base
        norm_dir = np.linalg.norm(direction)
        
        if norm_dir < 1e-10:
            return np.zeros_like(base)
        
        return theta * direction / norm_dir
    
    def project(self, x: NDArray) -> NDArray:
        """Проекция на сферу (нормализация)"""
        return self._normalize(x)
    
    def random_point(self, dim: int) -> NDArray:
        """Случайная точка на сфере (равномерное распределение)"""
        x = np.random.randn(dim)
        return self._normalize(x)
    
    def _normalize(self, x: NDArray) -> NDArray:
        """Нормализация вектора"""
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            x = np.random.randn(*x.shape)
            norm = np.linalg.norm(x)
        return x / norm
    
    def _ensure_normalized(self, x: NDArray) -> NDArray:
        """Гарантирует что вектор нормализован"""
        norm = np.linalg.norm(x)
        if abs(norm - 1.0) > 1e-6:
            return x / (norm + 1e-10)
        return x


# ============================================================
# УДОБНЫЕ ФУНКЦИИ (используют Sphere по умолчанию)
# ============================================================

_default_sphere = Sphere()


def d_geo(x: NDArray, y: NDArray) -> float:
    """
    Геодезическое расстояние между двумя shapes
    
    Args:
        x, y: нормализованные вектора
        
    Returns:
        Угол в радианах
    """
    return _default_sphere.distance(x, y)


def d_geo_deg(x: NDArray, y: NDArray) -> float:
    """Геодезическое расстояние в градусах"""
    return np.degrees(d_geo(x, y))


def centroid(shapes: NDArray) -> NDArray:
    """
    Центроид множества shapes
    
    Args:
        shapes: [N, dim] массив shapes
        
    Returns:
        Центроид на сфере
    """
    return _default_sphere.centroid(shapes)


def slerp(x: NDArray, y: NDArray, t: float) -> NDArray:
    """
    Интерполяция между shapes
    
    Args:
        x, y: shapes
        t: параметр [0, 1]
        
    Returns:
        Промежуточный shape
    """
    return _default_sphere.slerp(x, y, t)


def frechet_mean(shapes: NDArray) -> NDArray:
    """Точное среднее Фреше"""
    return _default_sphere.frechet_mean(shapes)


def distance_matrix(shapes: NDArray) -> NDArray:
    """
    Матрица попарных расстояний
    
    Args:
        shapes: [N, dim]
        
    Returns:
        [N, N] матрица расстояний
    """
    n = len(shapes)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = d_geo(shapes[i], shapes[j])
            D[i, j] = d
            D[j, i] = d
    return D
