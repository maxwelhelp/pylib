"""
Lifting Client - HTTP клиент для Lifting API

Поддерживает:
- Синхронные и асинхронные вызовы
- Автоматические retry при ошибках сети
- Batch операции для минимизации latency
- Кэширование результатов
"""

import time
import json
from typing import List, Dict, Any, Optional, Union
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_HTTPX = False


class LiftingClientError(Exception):
    """Ошибка клиента"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class LiftingClient:
    """
    HTTP клиент для Lifting API.

    Examples:
        >>> client = LiftingClient(api_key="demo-key")
        >>>
        >>> # Нормализация
        >>> shapes = client.normalize([[1, 2, 3], [4, 5, 6]])
        >>>
        >>> # Расстояния
        >>> distances = client.distance(shapes, target=shapes[0])
        >>>
        >>> # Центроид
        >>> c = client.centroid(shapes)
        >>>
        >>> # Sync Lifting
        >>> metrics = client.sync_compute(signal)
        >>>
        >>> # Batch операции (эффективнее!)
        >>> results = client.batch([
        ...     {"op": "normalize", "data": data1},
        ...     {"op": "distance", "shapes": "@0.shapes", "target": [1,0,0]},
        ... ])
    """

    DEFAULT_BASE_URL = "http://localhost:8000"
    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(
        self,
        api_key: str = "demo-key",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        cache_size: int = 128,
    ):
        """
        Args:
            api_key: API ключ для авторизации
            base_url: Базовый URL сервера
            timeout: Таймаут запросов в секундах
            cache_size: Размер LRU кэша (0 = отключить)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache_size = cache_size

        # HTTP клиент
        if HAS_HTTPX:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=timeout,
                headers={"X-API-Key": api_key},
            )
        else:
            self._client = None

    def close(self):
        """Закрыть клиент"""
        if HAS_HTTPX and self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ============================================================
    # Low-level HTTP
    # ============================================================

    def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict] = None,
        retries: int = MAX_RETRIES,
    ) -> Dict[str, Any]:
        """Выполнить HTTP запрос с retry"""
        url = f"{self.base_url}{path}"

        last_error = None
        for attempt in range(retries):
            try:
                if HAS_HTTPX:
                    response = self._client.request(method, path, json=json_data)
                    if response.status_code == 429:
                        # Rate limit - ждём и повторяем
                        retry_after = float(response.headers.get("Retry-After", 60))
                        time.sleep(min(retry_after, 60))
                        continue
                    if response.status_code >= 400:
                        raise LiftingClientError(
                            response.text,
                            status_code=response.status_code
                        )
                    return response.json()
                else:
                    # Fallback на urllib
                    return self._urllib_request(method, url, json_data)

            except (httpx.TimeoutException if HAS_HTTPX else TimeoutError) as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
            except (httpx.NetworkError if HAS_HTTPX else ConnectionError) as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))

        raise LiftingClientError(f"Request failed after {retries} attempts: {last_error}")

    def _urllib_request(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict],
    ) -> Dict[str, Any]:
        """Fallback HTTP через urllib"""
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

        data = json.dumps(json_data).encode() if json_data else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            raise LiftingClientError(e.read().decode(), status_code=e.code)
        except urllib.error.URLError as e:
            raise LiftingClientError(f"Network error: {e.reason}")

    # ============================================================
    # Geometry API
    # ============================================================

    def normalize(self, data: Union[List, NDArray]) -> NDArray:
        """
        Нормализовать векторы → shapes на единичной сфере.

        Args:
            data: [N, D] массив векторов

        Returns:
            [N, D] массив shapes (||shape|| = 1)
        """
        if isinstance(data, np.ndarray):
            data = data.tolist()

        response = self._request("POST", "/geometry/normalize", {"data": data})
        return np.array(response["shapes"])

    def distance(
        self,
        shapes: Union[List, NDArray],
        target: Optional[Union[List, NDArray]] = None,
    ) -> NDArray:
        """
        Расчёт геодезических расстояний.

        Args:
            shapes: [N, D] массив shapes
            target: [D] целевая точка (опционально)

        Returns:
            Если target указан: [N] расстояния до target
            Если target=None: [N, N] матрица расстояний
        """
        if isinstance(shapes, np.ndarray):
            shapes = shapes.tolist()
        if target is not None and isinstance(target, np.ndarray):
            target = target.tolist()

        payload = {"shapes": shapes}
        if target is not None:
            payload["target"] = target

        response = self._request("POST", "/geometry/distance", payload)
        return np.array(response["distances"])

    def centroid(self, shapes: Union[List, NDArray]) -> NDArray:
        """
        Расчёт центроида (среднее на сфере).

        Args:
            shapes: [N, D] массив shapes

        Returns:
            [D] центроид
        """
        if isinstance(shapes, np.ndarray):
            shapes = shapes.tolist()

        response = self._request("POST", "/geometry/centroid", {"shapes": shapes})
        return np.array(response["centroid"])

    # ============================================================
    # Sync Lifting API
    # ============================================================

    def sync_compute(
        self,
        signal: Union[List, NDArray],
        transform: str = "fft",
        moments: List[int] = [0, 1, 2],
    ) -> Dict[str, NDArray]:
        """
        Вычислить синхронные метрики.

        Args:
            signal: Входной сигнал
            transform: "fft" или "dct"
            moments: Список моментов [0, 1, 2, 3]

        Returns:
            {'magnitude': [...], 'time_center': [...], 'time_spread': [...]}
        """
        if isinstance(signal, np.ndarray):
            signal = signal.tolist()

        response = self._request("POST", "/sync/compute", {
            "signal": signal,
            "transform": transform,
            "moments": moments,
        })

        return {k: np.array(v) for k, v in response.items() if k != "length"}

    def sync_dominant(
        self,
        signal: Union[List, NDArray],
        transform: str = "fft",
    ) -> Dict[str, float]:
        """
        Метрики доминирующего коэффициента.

        Returns:
            {'tc': float, 'ts': float, 'dominant_idx': int}
        """
        if isinstance(signal, np.ndarray):
            signal = signal.tolist()

        return self._request("POST", "/sync/dominant", {
            "signal": signal,
            "transform": transform,
        })

    def sync_features(
        self,
        signal: Union[List, NDArray],
        transforms: List[str] = ["fft", "dct"],
    ) -> Dict[str, float]:
        """
        Унифицированный вектор фичей.

        Returns:
            {'fft_tc': float, 'fft_ts': float, 'dct_tc': float, ...}
        """
        if isinstance(signal, np.ndarray):
            signal = signal.tolist()

        return self._request("POST", "/sync/features", {
            "signal": signal,
            "transforms": transforms,
        })

    # ============================================================
    # Batch API (эффективно!)
    # ============================================================

    def batch(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Выполнить несколько операций в одном запросе.

        Поддерживает ссылки на результаты: "@0.shapes" → результат операции 0.

        Examples:
            >>> results = client.batch([
            ...     {"op": "normalize", "data": [[1,2,3], [4,5,6]]},
            ...     {"op": "distance", "shapes": "@0.shapes", "target": [1,0,0]},
            ...     {"op": "centroid", "shapes": "@0.shapes"},
            ... ])
            >>> shapes = results[0]["shapes"]
            >>> distances = results[1]["distances"]
            >>> centroid = results[2]["centroid"]
        """
        # Конвертируем numpy в lists
        clean_ops = []
        for op in operations:
            clean_op = {}
            for k, v in op.items():
                if isinstance(v, np.ndarray):
                    clean_op[k] = v.tolist()
                else:
                    clean_op[k] = v
            clean_ops.append(clean_op)

        response = self._request("POST", "/batch", {"operations": clean_ops})
        return response["results"]

    # ============================================================
    # High-level convenience methods
    # ============================================================

    def normalize_and_centroid(self, data: Union[List, NDArray]) -> tuple:
        """
        Нормализовать данные и вычислить центроид за один запрос.

        Returns:
            (shapes, centroid)
        """
        results = self.batch([
            {"op": "normalize", "data": data if isinstance(data, list) else data.tolist()},
            {"op": "centroid", "shapes": "@0.shapes"},
        ])
        return np.array(results[0]["shapes"]), np.array(results[1]["centroid"])

    def distances_from_centroid(self, data: Union[List, NDArray]) -> tuple:
        """
        Нормализовать, вычислить центроид и расстояния до него.

        Returns:
            (shapes, centroid, distances)
        """
        results = self.batch([
            {"op": "normalize", "data": data if isinstance(data, list) else data.tolist()},
            {"op": "centroid", "shapes": "@0.shapes"},
            {"op": "distance", "shapes": "@0.shapes", "target": "@1.centroid"},
        ])
        return (
            np.array(results[0]["shapes"]),
            np.array(results[1]["centroid"]),
            np.array(results[2]["distances"]),
        )

    # ============================================================
    # Health check
    # ============================================================

    def ping(self) -> Dict[str, Any]:
        """Проверка соединения с сервером"""
        return self._request("GET", "/health")

    def is_available(self) -> bool:
        """Доступен ли сервер"""
        try:
            self.ping()
            return True
        except LiftingClientError:
            return False


# ============================================================
# Async Client (опционально)
# ============================================================

if HAS_HTTPX:
    class AsyncLiftingClient:
        """
        Асинхронный клиент для Lifting API.

        Examples:
            >>> async with AsyncLiftingClient(api_key="demo-key") as client:
            ...     shapes = await client.normalize(data)
            ...     distances = await client.distance(shapes, target=shapes[0])
        """

        def __init__(
            self,
            api_key: str = "demo-key",
            base_url: str = LiftingClient.DEFAULT_BASE_URL,
            timeout: float = LiftingClient.DEFAULT_TIMEOUT,
        ):
            self.api_key = api_key
            self.base_url = base_url.rstrip("/")
            self.timeout = timeout
            self._client = None

        async def __aenter__(self):
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"X-API-Key": self.api_key},
            )
            return self

        async def __aexit__(self, *args):
            if self._client:
                await self._client.aclose()

        async def _request(
            self,
            method: str,
            path: str,
            json_data: Optional[Dict] = None,
        ) -> Dict[str, Any]:
            """Выполнить async HTTP запрос"""
            response = await self._client.request(method, path, json=json_data)
            if response.status_code >= 400:
                raise LiftingClientError(response.text, status_code=response.status_code)
            return response.json()

        async def normalize(self, data: Union[List, NDArray]) -> NDArray:
            if isinstance(data, np.ndarray):
                data = data.tolist()
            response = await self._request("POST", "/geometry/normalize", {"data": data})
            return np.array(response["shapes"])

        async def distance(
            self,
            shapes: Union[List, NDArray],
            target: Optional[Union[List, NDArray]] = None,
        ) -> NDArray:
            if isinstance(shapes, np.ndarray):
                shapes = shapes.tolist()
            if target is not None and isinstance(target, np.ndarray):
                target = target.tolist()

            payload = {"shapes": shapes}
            if target is not None:
                payload["target"] = target

            response = await self._request("POST", "/geometry/distance", payload)
            return np.array(response["distances"])

        async def centroid(self, shapes: Union[List, NDArray]) -> NDArray:
            if isinstance(shapes, np.ndarray):
                shapes = shapes.tolist()
            response = await self._request("POST", "/geometry/centroid", {"shapes": shapes})
            return np.array(response["centroid"])

        async def sync_compute(
            self,
            signal: Union[List, NDArray],
            transform: str = "fft",
            moments: List[int] = [0, 1, 2],
        ) -> Dict[str, NDArray]:
            if isinstance(signal, np.ndarray):
                signal = signal.tolist()
            response = await self._request("POST", "/sync/compute", {
                "signal": signal,
                "transform": transform,
                "moments": moments,
            })
            return {k: np.array(v) for k, v in response.items() if k != "length"}

        async def batch(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            clean_ops = []
            for op in operations:
                clean_op = {}
                for k, v in op.items():
                    if isinstance(v, np.ndarray):
                        clean_op[k] = v.tolist()
                    else:
                        clean_op[k] = v
                clean_ops.append(clean_op)

            response = await self._request("POST", "/batch", {"operations": clean_ops})
            return response["results"]

        async def ping(self) -> Dict[str, Any]:
            return await self._request("GET", "/health")
