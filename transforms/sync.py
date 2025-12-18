"""
Synchronous Lifting - КЛЮЧЕВОЙ МОДУЛЬ

Синхронный лифтинг через временные моменты:
    T(x)      → ЧТО (коэффициенты преобразования)
    T(x·t)    → ГДЕ (time_center - где событие)
    T(x·t²)   → КАК ДОЛГО (time_spread - длительность)
    T(x·t³)   → НАПРАВЛЕНИЕ (skewness)

Это даёт ИНТЕРПРЕТИРУЕМЫЕ метрики для каждого коэффициента.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Callable, Optional, Any, Tuple
from functools import lru_cache


# Глобальный кэш для t_powers (избегает пересоздания для одинаковых N)
@lru_cache(maxsize=16)
def _get_time_axis(N: int) -> NDArray:
    """Кэшированная временная ось [0, 1]"""
    return np.linspace(0, 1, N)


@lru_cache(maxsize=64)
def _get_t_power(N: int, k: int) -> NDArray:
    """Кэшированная степень временной оси t^k"""
    return _get_time_axis(N) ** k


class SyncLifting:
    """
    Синхронный лифтинг — вычисление time_center, time_spread через моменты.
    
    Формула:
        time_center[k] = |T(x·t)[k]| / |T(x)[k]|
        time_spread[k] = sqrt(|T(x·t²)[k]|/|T(x)[k]| - time_center[k]²)
    
    Где T — любое преобразование (FFT, DCT, Wavelet, ...).
    
    Examples:
        >>> sync = SyncLifting(moments=[0, 1, 2])
        >>> metrics = sync.compute(signal, np.fft.rfft)
        >>> print(metrics['time_center'])  # где энергия каждой частоты
        >>> print(metrics['time_spread'])  # spread каждой частоты
    """
    
    def __init__(self, N: Optional[int] = None, moments: List[int] = [0, 1, 2]):
        """
        Args:
            N: длина сигнала (можно не указывать, определится автоматически)
            moments: какие моменты использовать [0, 1, 2] = стандартный набор
        """
        self.N = N
        self.moments = moments
        self._t = None
        self._t_powers = None
    
    def _init_time(self, N: int):
        """Инициализация временной оси (использует глобальный кэш)"""
        if self._t is None or len(self._t) != N:
            self._t = _get_time_axis(N)
            # Используем кэшированные степени
            self._t_powers = {k: _get_t_power(N, k) for k in self.moments}
    
    def compute(self, signal: NDArray, transform: Callable) -> Dict[str, Any]:
        """
        Вычислить синхронные метрики для преобразования.
        
        Args:
            signal: входной сигнал
            transform: функция преобразования (np.fft.rfft, dct, ...)
            
        Returns:
            {
                'magnitude': |T(x)|,
                'time_center': tc[k] для каждого коэффициента,
                'time_spread': ts[k] для каждого коэффициента,
                'raw': {k: T(x·t^k)} для всех k
            }
        """
        signal = np.asarray(signal)
        N = len(signal)
        self._init_time(N)
        
        # T(x · t^k) для каждого k
        transforms = {}
        for k in self.moments:
            weighted = signal * self._t_powers[k]
            transforms[k] = transform(weighted)
        
        # Magnitude
        T0 = transforms[0]
        magnitude = np.abs(T0)
        magnitude_safe = np.maximum(magnitude, 1e-10)
        
        result = {
            'magnitude': magnitude,
            'raw': transforms,
        }
        
        # Time center: E[t] = |T(x·t)| / |T(x)|
        if 1 in transforms:
            T1 = transforms[1]
            time_center = np.abs(T1) / magnitude_safe
            time_center = np.clip(time_center, 0, 1)
            result['time_center'] = time_center
        
        # Time spread: sqrt(E[t²] - E[t]²)
        if 2 in transforms and 'time_center' in result:
            T2 = transforms[2]
            variance = np.abs(T2) / magnitude_safe - result['time_center'] ** 2
            time_spread = np.sqrt(np.maximum(variance, 0))
            result['time_spread'] = time_spread
        
        # Skewness (если есть момент 3)
        if 3 in transforms and 'time_center' in result and 'time_spread' in result:
            T3 = transforms[3]
            tc = result['time_center']
            ts = result['time_spread']
            T2 = transforms[2]
            
            m3 = np.abs(T3) / magnitude_safe - 3 * tc * (np.abs(T2) / magnitude_safe) + 2 * tc ** 3
            skewness = m3 / (ts ** 3 + 1e-10)
            result['skewness'] = np.clip(skewness, -10, 10)
        
        return result
    
    def compute_envelope_metrics(self, envelope: NDArray) -> Dict[str, float]:
        """
        Метрики для огибающей (уже в временной области).
        
        Использовать когда есть envelope (Hilbert, Teager, ...).
        
        Args:
            envelope: огибающая сигнала (неотрицательная)
            
        Returns:
            {'time_center': float, 'time_spread': float}
        """
        envelope = np.asarray(envelope)
        N = len(envelope)
        self._init_time(N)
        
        # Нормализуем как распределение
        env_norm = envelope / (envelope.sum() + 1e-10)
        
        # E[t]
        time_center = float(np.sum(self._t * env_norm))
        
        # Var[t]
        variance = float(np.sum((self._t - time_center) ** 2 * env_norm))
        time_spread = np.sqrt(variance)
        
        return {
            'time_center': time_center,
            'time_spread': time_spread,
        }
    
    def dominant_metrics(self, signal: NDArray, transform: Callable) -> Dict[str, float]:
        """
        Метрики для доминирующего коэффициента.
        
        Удобно когда нужна одна пара (tc, ts) вместо массива.
        
        Returns:
            {'tc': float, 'ts': float, 'dominant_idx': int}
        """
        metrics = self.compute(signal, transform)
        
        # Доминирующий коэффициент (максимальная magnitude, пропускаем DC)
        mag = metrics['magnitude']
        if len(mag) > 1:
            dom_idx = np.argmax(mag[1:]) + 1
        else:
            dom_idx = 0
        
        result = {'dominant_idx': dom_idx}
        
        if 'time_center' in metrics:
            result['tc'] = float(metrics['time_center'][dom_idx])
        
        if 'time_spread' in metrics:
            result['ts'] = float(metrics['time_spread'][dom_idx])
        
        return result


def sync_fft(signal: NDArray, moments: List[int] = [0, 1, 2]) -> Dict[str, Any]:
    """
    Синхронный FFT — удобная обёртка
    
    Examples:
        >>> metrics = sync_fft(signal)
        >>> tc = metrics['time_center']  # где энергия каждой частоты
    """
    sync = SyncLifting(moments=moments)
    return sync.compute(signal, np.fft.rfft)


def sync_dct(signal: NDArray, moments: List[int] = [0, 1, 2]) -> Dict[str, Any]:
    """Синхронный DCT"""
    from scipy.fftpack import dct
    sync = SyncLifting(moments=moments)
    return sync.compute(signal, lambda x: dct(x, type=2, norm='ortho'))


def get_unified_features(signal: NDArray, 
                         transforms: List[str] = ['fft', 'dct'],
                         moments: List[int] = [0, 1, 2]) -> Dict[str, float]:
    """
    Унифицированный вектор фичей от нескольких преобразований.
    
    Args:
        signal: входной сигнал
        transforms: список преобразований ['fft', 'dct', 'hilbert', 'teager']
        moments: моменты для sync lifting
        
    Returns:
        Словарь {'{transform}_{metric}': value}
    """
    from scipy.fftpack import dct as scipy_dct
    from .analytic import hilbert, teager_energy, envelope
    
    sync = SyncLifting(moments=moments)
    features = {}
    
    for name in transforms:
        if name == 'fft':
            m = sync.dominant_metrics(signal, np.fft.rfft)
            features['fft_tc'] = m.get('tc', 0.5)
            features['fft_ts'] = m.get('ts', 0.0)
            
        elif name == 'dct':
            m = sync.dominant_metrics(signal, lambda x: scipy_dct(x, type=2, norm='ortho'))
            features['dct_tc'] = m.get('tc', 0.5)
            features['dct_ts'] = m.get('ts', 0.0)
            
        elif name == 'hilbert':
            env = envelope(signal)
            m = sync.compute_envelope_metrics(env)
            features['hil_tc'] = m['time_center']
            features['hil_ts'] = m['time_spread']
            
        elif name == 'teager':
            teo = teager_energy(signal)
            m = sync.compute_envelope_metrics(teo)
            features['teo_tc'] = m['time_center']
            features['teo_ts'] = m['time_spread']
    
    return features
