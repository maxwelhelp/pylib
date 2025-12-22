"""
Backend Abstraction - переключение между локальным и удалённым API.

Позволяет tasks/ работать как с локальным core (разработка),
так и с удалённым API (production, когда core скрыт).

Использование:
    >>> from tasks.backend import get_backend, set_backend
    >>>
    >>> # По умолчанию - локальный core
    >>> backend = get_backend()
    >>> shapes = backend.normalize(data)
    >>>
    >>> # Переключить на удалённый API
    >>> from client import LiftingClient
    >>> set_backend(LiftingClient(api_key="your-key"))
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, List, Dict, Any, Protocol
from abc import ABC, abstractmethod


class Backend(Protocol):
    """Протокол backend для геометрических и sync операций."""

    def normalize(self, data: NDArray) -> NDArray:
        """Нормализовать векторы → shapes"""
        ...

    def d_geo(self, shape1: NDArray, shape2: NDArray) -> float:
        """Геодезическое расстояние между двумя shapes"""
        ...

    def d_geo_batch(self, shapes: NDArray, target: NDArray) -> NDArray:
        """Расстояния от shapes до target"""
        ...

    def centroid(self, shapes: NDArray) -> NDArray:
        """Центроид (среднее на сфере)"""
        ...

    def sync_compute(self, signal: NDArray, transform: str, moments: List[int]) -> Dict[str, Any]:
        """SyncLifting compute"""
        ...

    def sync_dominant(self, signal: NDArray, transform: str) -> Dict[str, float]:
        """SyncLifting dominant metrics"""
        ...

    def sync_features(self, signal: NDArray, transforms: List[str]) -> Dict[str, float]:
        """Unified sync features"""
        ...

    def d_eff(self, shapes: NDArray, threshold: float) -> int:
        """Effective dimensionality"""
        ...


class LocalBackend:
    """
    Локальный backend - использует core напрямую.

    Для разработки и тестирования.
    """

    def __init__(self):
        # Ленивый импорт core
        self._core = None

    def _get_core(self):
        if self._core is None:
            try:
                from ..core import normalize, d_geo, d_geo_batch, centroid
                self._core = {
                    'normalize': normalize,
                    'd_geo': d_geo,
                    'd_geo_batch': d_geo_batch,
                    'centroid': centroid,
                }
            except ImportError:
                from core import normalize, d_geo, d_geo_batch, centroid
                self._core = {
                    'normalize': normalize,
                    'd_geo': d_geo,
                    'd_geo_batch': d_geo_batch,
                    'centroid': centroid,
                }
        return self._core

    def normalize(self, data: NDArray) -> NDArray:
        return self._get_core()['normalize'](data)

    def d_geo(self, shape1: NDArray, shape2: NDArray) -> float:
        return self._get_core()['d_geo'](shape1, shape2)

    def d_geo_batch(self, shapes: NDArray, target: NDArray) -> NDArray:
        return self._get_core()['d_geo_batch'](shapes, target)

    def centroid(self, shapes: NDArray) -> NDArray:
        return self._get_core()['centroid'](shapes)

    def sync_compute(self, signal: NDArray, transform: str = "fft", moments: List[int] = [0, 1, 2]) -> Dict[str, Any]:
        """SyncLifting через локальный transforms"""
        try:
            from ..transforms.sync import SyncLifting
        except ImportError:
            from transforms.sync import SyncLifting

        sync = SyncLifting(moments=moments)
        if transform == "fft":
            return sync.compute(signal, np.fft.rfft)
        elif transform == "dct":
            from scipy.fftpack import dct
            return sync.compute(signal, lambda x: dct(x, type=2, norm='ortho'))
        else:
            return sync.compute(signal, np.fft.rfft)

    def sync_dominant(self, signal: NDArray, transform: str = "fft") -> Dict[str, float]:
        """Dominant metrics через локальный transforms"""
        try:
            from ..transforms.sync import SyncLifting
        except ImportError:
            from transforms.sync import SyncLifting

        sync = SyncLifting(moments=[0, 1, 2])
        if transform == "fft":
            return sync.dominant_metrics(signal, np.fft.rfft)
        elif transform == "dct":
            from scipy.fftpack import dct
            return sync.dominant_metrics(signal, lambda x: dct(x, type=2, norm='ortho'))
        else:
            return sync.dominant_metrics(signal, np.fft.rfft)

    def sync_features(self, signal: NDArray, transforms: List[str] = ["fft", "dct"]) -> Dict[str, float]:
        """Unified features через локальный transforms"""
        try:
            from ..transforms.sync import get_unified_features
        except ImportError:
            from transforms.sync import get_unified_features
        return get_unified_features(signal, transforms)

    def d_eff(self, shapes: NDArray, threshold: float = 0.95) -> int:
        """Effective dimensionality через локальный core"""
        try:
            from ..core.metrics import d_eff
        except ImportError:
            from core.metrics import d_eff
        return d_eff(shapes, threshold)


class RemoteBackend:
    """
    Удалённый backend - использует LiftingClient для API.

    Для production, когда core скрыт на сервере.
    """

    def __init__(self, client):
        """
        Args:
            client: LiftingClient instance
        """
        self.client = client

    def normalize(self, data: NDArray) -> NDArray:
        return self.client.normalize(data)

    def d_geo(self, shape1: NDArray, shape2: NDArray) -> float:
        # Одиночное расстояние через batch
        distances = self.client.distance(
            shapes=shape1.reshape(1, -1),
            target=shape2
        )
        return float(distances[0])

    def d_geo_batch(self, shapes: NDArray, target: NDArray) -> NDArray:
        return self.client.distance(shapes, target=target)

    def centroid(self, shapes: NDArray) -> NDArray:
        return self.client.centroid(shapes)

    def sync_compute(self, signal: NDArray, transform: str = "fft", moments: List[int] = [0, 1, 2]) -> Dict[str, Any]:
        """SyncLifting через API"""
        return self.client.sync_compute(signal, transform, moments)

    def sync_dominant(self, signal: NDArray, transform: str = "fft") -> Dict[str, float]:
        """Dominant metrics через API"""
        return self.client.sync_dominant(signal, transform)

    def sync_features(self, signal: NDArray, transforms: List[str] = ["fft", "dct"]) -> Dict[str, float]:
        """Unified features через API"""
        return self.client.sync_features(signal, transforms)

    def d_eff(self, shapes: NDArray, threshold: float = 0.95) -> int:
        """d_eff через API"""
        return self.client.d_eff(shapes, threshold)


# ============================================================
# Global Backend State
# ============================================================

_current_backend: Optional[Backend] = None


def get_backend() -> Backend:
    """
    Получить текущий backend.

    По умолчанию возвращает LocalBackend.

    Returns:
        Backend instance
    """
    global _current_backend
    if _current_backend is None:
        _current_backend = LocalBackend()
    return _current_backend


def set_backend(backend: Union[Backend, 'LiftingClient', None]) -> None:
    """
    Установить backend.

    Args:
        backend: Backend instance, LiftingClient, или None (сброс на локальный)

    Examples:
        >>> # Использовать локальный core
        >>> set_backend(None)
        >>>
        >>> # Использовать удалённый API
        >>> from client import LiftingClient
        >>> set_backend(LiftingClient(api_key="your-key"))
        >>>
        >>> # Или напрямую RemoteBackend
        >>> set_backend(RemoteBackend(client))
    """
    global _current_backend

    if backend is None:
        _current_backend = LocalBackend()
    elif isinstance(backend, (LocalBackend, RemoteBackend)):
        _current_backend = backend
    else:
        # Предполагаем LiftingClient
        _current_backend = RemoteBackend(backend)


def use_remote(api_key: str = "demo-key", base_url: str = "http://localhost:8000"):
    """
    Переключить на удалённый backend.

    Convenience function.

    Examples:
        >>> use_remote(api_key="your-key", base_url="https://api.example.com")
    """
    try:
        from client import LiftingClient
    except ImportError:
        from ..client import LiftingClient

    client = LiftingClient(api_key=api_key, base_url=base_url)
    set_backend(client)


def use_local():
    """
    Переключить на локальный backend.

    Examples:
        >>> use_local()
    """
    set_backend(None)


# ============================================================
# Context manager for temporary backend switch
# ============================================================

class backend_context:
    """
    Временное переключение backend.

    Examples:
        >>> with backend_context(LiftingClient(api_key="test")):
        ...     # Здесь используется remote backend
        ...     detector.fit(shapes)
        >>> # Здесь снова локальный
    """

    def __init__(self, backend: Union[Backend, 'LiftingClient', None]):
        self.new_backend = backend
        self.old_backend = None

    def __enter__(self):
        global _current_backend
        self.old_backend = _current_backend
        set_backend(self.new_backend)
        return get_backend()

    def __exit__(self, *args):
        global _current_backend
        _current_backend = self.old_backend
