"""
Wavelet Transforms - CWT, DWT

Многомасштабный анализ сигналов.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List


def cwt(signal: NDArray, 
        scales: Optional[NDArray] = None,
        wavelet: str = 'morlet') -> NDArray:
    """
    Continuous Wavelet Transform
    
    Args:
        signal: входной сигнал
        scales: масштабы (по умолчанию логарифмическая сетка)
        wavelet: тип вейвлета ('morlet', 'mexican_hat')
        
    Returns:
        [n_scales, n_time] коэффициенты (magnitude)
    """
    try:
        import pywt
        
        if scales is None:
            scales = np.arange(1, min(64, len(signal) // 4))
        
        if wavelet == 'morlet':
            wavelet_name = 'morl'
        elif wavelet == 'mexican_hat':
            wavelet_name = 'mexh'
        else:
            wavelet_name = wavelet
        
        coeffs, _ = pywt.cwt(signal, scales, wavelet_name)
        return np.abs(coeffs)
        
    except ImportError:
        # Fallback без pywt
        return _cwt_simple(signal, scales, wavelet)


def _cwt_simple(signal: NDArray, 
                scales: Optional[NDArray] = None,
                wavelet: str = 'morlet') -> NDArray:
    """Простая реализация CWT без pywt"""
    n = len(signal)
    
    if scales is None:
        scales = 2 ** np.linspace(0, np.log2(n/4), 32)
    
    t = np.arange(n)
    coeffs = []
    
    for scale in scales:
        wavelet_t = (t - n/2) / scale
        
        if wavelet == 'morlet':
            omega0 = 5.0
            psi = np.exp(1j * omega0 * wavelet_t) * np.exp(-wavelet_t**2 / 2)
        elif wavelet == 'mexican_hat':
            psi = (1 - wavelet_t**2) * np.exp(-wavelet_t**2 / 2)
        else:
            psi = np.exp(-wavelet_t**2 / 2)
        
        conv = np.convolve(signal, psi, mode='same') / np.sqrt(scale)
        coeffs.append(conv)
    
    return np.abs(np.array(coeffs))


def dwt(signal: NDArray, wavelet: str = 'db4', level: int = 5) -> List[NDArray]:
    """
    Discrete Wavelet Transform
    
    Args:
        signal: входной сигнал
        wavelet: тип вейвлета ('haar', 'db4', 'sym4')
        level: количество уровней разложения
        
    Returns:
        [approx, detail_1, detail_2, ..., detail_n]
    """
    try:
        import pywt
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        return coeffs
    except ImportError:
        # Fallback: только Haar
        return _dwt_haar(signal, level)


def _dwt_haar(signal: NDArray, level: int = 5) -> List[NDArray]:
    """Простой Haar DWT"""
    lo = np.array([1, 1]) / np.sqrt(2)
    hi = np.array([1, -1]) / np.sqrt(2)
    
    coeffs = []
    approx = signal.copy()
    
    for _ in range(level):
        n = len(approx)
        if n < 2:
            break
        
        approx_new = np.convolve(approx, lo, mode='same')[::2]
        detail = np.convolve(approx, hi, mode='same')[::2]
        
        coeffs.append(detail)
        approx = approx_new
    
    return [approx] + coeffs[::-1]


def wavelet_energy(signal: NDArray, 
                   scales: Optional[NDArray] = None,
                   wavelet: str = 'morlet') -> NDArray:
    """
    Энергия по масштабам (scalogram)
    
    Returns:
        [n_scales] энергия на каждом масштабе
    """
    coeffs = cwt(signal, scales, wavelet)
    return np.sum(coeffs ** 2, axis=1)


def wavelet_features(signal: NDArray,
                     n_scales: int = 8,
                     wavelet: str = 'morlet') -> NDArray:
    """
    Вейвлет-фичи: энергия + статистики по масштабам
    
    Returns:
        Вектор фичей
    """
    scales = 2 ** np.linspace(0, np.log2(len(signal) / 4), n_scales)
    coeffs = cwt(signal, scales, wavelet)
    
    features = []
    
    for s_idx in range(len(scales)):
        c = coeffs[s_idx]
        features.extend([
            np.sum(c ** 2),      # энергия
            np.mean(c),          # среднее
            np.std(c),           # std
            np.max(np.abs(c)),   # max
        ])
    
    return np.array(features)
