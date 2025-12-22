"""
Synchronous Lifting - CLIENT VERSION

Этот модуль использует API для вычислений.
Математика скрыта на сервере.

Использование:
    >>> from transforms import SyncLifting, sync_fft
    >>> sync = SyncLifting()
    >>> metrics = sync.compute(signal, "fft")
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Any

# Получаем backend для вызовов API
try:
    from ..tasks.backend import get_backend
except ImportError:
    from tasks.backend import get_backend


class SyncLifting:
    """
    Синхронный лифтинг через API.

    Вычисляет time_center, time_spread для каждого коэффициента.

    Examples:
        >>> sync = SyncLifting(moments=[0, 1, 2])
        >>> metrics = sync.compute(signal, "fft")
        >>> print(metrics['time_center'])
    """

    def __init__(self, N: int = None, moments: List[int] = [0, 1, 2]):
        """
        Args:
            N: длина сигнала (опционально)
            moments: какие моменты использовать
        """
        self.N = N
        self.moments = moments

    def compute(self, signal: NDArray, transform) -> Dict[str, Any]:
        """
        Вычислить синхронные метрики.

        Args:
            signal: входной сигнал
            transform: "fft", "dct" или callable (для совместимости)

        Returns:
            {'magnitude': [...], 'time_center': [...], 'time_spread': [...]}
        """
        signal = np.asarray(signal)

        # Определяем тип преобразования
        if callable(transform):
            # Для совместимости со старым API
            transform_name = "fft"
            if hasattr(transform, '__name__'):
                if 'dct' in transform.__name__.lower():
                    transform_name = "dct"
        else:
            transform_name = str(transform)

        backend = get_backend()
        return backend.sync_compute(signal, transform_name, self.moments)

    def dominant_metrics(self, signal: NDArray, transform) -> Dict[str, float]:
        """
        Метрики для доминирующего коэффициента.

        Returns:
            {'tc': float, 'ts': float, 'dominant_idx': int}
        """
        signal = np.asarray(signal)

        # Определяем тип преобразования
        if callable(transform):
            transform_name = "fft"
            if hasattr(transform, '__name__'):
                if 'dct' in transform.__name__.lower():
                    transform_name = "dct"
        else:
            transform_name = str(transform)

        backend = get_backend()
        return backend.sync_dominant(signal, transform_name)

    def compute_envelope_metrics(self, envelope: NDArray) -> Dict[str, float]:
        """
        Метрики для огибающей.

        Вычисляется локально (не секретная математика).
        """
        envelope = np.asarray(envelope)
        N = len(envelope)
        t = np.linspace(0, 1, N)

        env_norm = envelope / (envelope.sum() + 1e-10)
        time_center = float(np.sum(t * env_norm))
        variance = float(np.sum((t - time_center) ** 2 * env_norm))
        time_spread = np.sqrt(variance)

        return {
            'time_center': time_center,
            'time_spread': time_spread,
        }


# ============================================================
# Convenience functions
# ============================================================

_sync_cache: Dict[tuple, SyncLifting] = {}


def get_sync(moments: List[int] = [0, 1, 2]) -> SyncLifting:
    """Получить кэшированный SyncLifting."""
    key = tuple(moments)
    if key not in _sync_cache:
        _sync_cache[key] = SyncLifting(moments=list(moments))
    return _sync_cache[key]


def clear_sync_cache() -> None:
    """Очистить кэш."""
    _sync_cache.clear()


def sync_fft(signal: NDArray, moments: List[int] = [0, 1, 2]) -> Dict[str, Any]:
    """Синхронный FFT через API."""
    return get_sync(moments).compute(signal, "fft")


def sync_dct(signal: NDArray, moments: List[int] = [0, 1, 2]) -> Dict[str, Any]:
    """Синхронный DCT через API."""
    return get_sync(moments).compute(signal, "dct")


def get_unified_features(
    signal: NDArray,
    transforms: List[str] = ['fft', 'dct'],
    moments: List[int] = [0, 1, 2]
) -> Dict[str, float]:
    """
    Унифицированный вектор фичей через API.

    Args:
        signal: входной сигнал
        transforms: список преобразований ['fft', 'dct']
        moments: моменты

    Returns:
        {'fft_tc': float, 'fft_ts': float, ...}
    """
    signal = np.asarray(signal)
    backend = get_backend()
    return backend.sync_features(signal, transforms)
