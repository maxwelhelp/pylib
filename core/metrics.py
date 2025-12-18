"""
Core Metrics - Метрики качества представлений

Содержит:
- d_eff: эффективная размерность
- kl_divergence: KL-дивергенция
- Связь KL и геодезического расстояния
"""

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import svd
from typing import List, Dict, Optional


def d_eff(representations: NDArray, threshold: float = 0.95) -> int:
    """
    Эффективная размерность данных
    
    d_eff = min{d : Σᵢ₌₁ᵈ σᵢ² ≥ threshold × Σᵢ σᵢ²}
    
    Показывает сколько "реальных" измерений в данных.
    Меньше d_eff = лучше сжатие.
    
    Args:
        representations: [N, dim] массив представлений
        threshold: доля объяснённой дисперсии (0.95 = 95%)
        
    Returns:
        Эффективная размерность
        
    Examples:
        >>> # Данные на прямой (1D структура)
        >>> d_eff(line_data)  # → 1
        
        >>> # Случайный шум
        >>> d_eff(noise)  # → большое число
    """
    reps = np.asarray(representations)
    
    if reps.ndim == 1:
        return 1
    
    if len(reps) < 2:
        return reps.shape[-1]
    
    # Центрирование
    centered = reps - np.mean(reps, axis=0)
    
    try:
        U, S, Vt = svd(centered, full_matrices=False)
    except:
        return reps.shape[-1]
    
    # Накопленная дисперсия
    var_explained = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
    
    # Минимальное d для достижения threshold
    d = np.searchsorted(var_explained, threshold) + 1
    return min(d, len(S))


def d_eff_spectrum(representations: NDArray) -> tuple:
    """
    Полный спектр сингулярных значений
    
    Returns:
        (singular_values, variance_explained)
    """
    reps = np.asarray(representations)
    centered = reps - np.mean(reps, axis=0)
    
    U, S, Vt = svd(centered, full_matrices=False)
    variance_explained = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
    
    return S, variance_explained


def d_eff_at_thresholds(representations: NDArray,
                        thresholds: List[float] = [0.90, 0.95, 0.99]) -> Dict[float, int]:
    """
    d_eff при разных порогах
    
    Returns:
        {threshold: d_eff}
    """
    return {t: d_eff(representations, threshold=t) for t in thresholds}


def kl_divergence(p: NDArray, q: NDArray, eps: float = 1e-10) -> float:
    """
    KL-дивергенция D_KL(p || q)
    
    D_KL = Σ p_i log(p_i / q_i)
    
    Args:
        p, q: вероятностные распределения
        eps: для численной стабильности
    """
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return float(np.sum(p * np.log(p / q)))


def kl_from_geodesic(geodesic_distance: float) -> float:
    """
    Оценка KL из геодезического расстояния
    
    Теорема: D_KL ≈ 2 × d_geo²
    
    Args:
        geodesic_distance: расстояние на сфере (в радианах)
    """
    return 2.0 * geodesic_distance ** 2


def geodesic_from_kl(kl: float) -> float:
    """
    Оценка геодезического расстояния из KL
    
    d_geo ≈ √(KL / 2)
    """
    return np.sqrt(kl / 2.0)


def variance_on_sphere(shapes: NDArray, center: Optional[NDArray] = None) -> float:
    """
    Дисперсия shapes на сфере
    
    σ² = (1/N) Σ d²(S_i, μ)
    
    Args:
        shapes: [N, dim] массив shapes
        center: центроид (если None, вычисляется)
    """
    from .geometry import centroid, d_geo
    
    if center is None:
        center = centroid(shapes)
    
    distances_sq = [d_geo(s, center)**2 for s in shapes]
    return float(np.mean(distances_sq))


def std_on_sphere(shapes: NDArray, center: Optional[NDArray] = None) -> float:
    """Стандартное отклонение на сфере"""
    return np.sqrt(variance_on_sphere(shapes, center))


def separation_ratio(shapes: NDArray, labels: NDArray) -> float:
    """
    Коэффициент разделения классов
    
    ratio = mean(inter-class distance) / mean(intra-class distance)
    
    Больше = лучше разделение.
    > 2.0: отличное разделение
    > 1.5: хорошее
    < 1.2: плохое
    """
    from .geometry import d_geo
    
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return float('inf')
    
    intra = []
    inter = []
    
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            d = d_geo(shapes[i], shapes[j])
            
            if labels[i] == labels[j]:
                intra.append(d)
            else:
                inter.append(d)
    
    if len(intra) == 0 or len(inter) == 0:
        return float('inf')
    
    return float(np.mean(inter) / (np.mean(intra) + 1e-10))
