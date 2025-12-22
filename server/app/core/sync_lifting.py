"""
Synchronous Lifting - СКРЫТАЯ МАТЕМАТИКА

Ключевой алгоритм библиотеки.
НЕ ПУБЛИКОВАТЬ.

Формулы:
    T(x)      → ЧТО (коэффициенты)
    T(x·t)    → ГДЕ (time_center)
    T(x·t²)   → КАК ДОЛГО (time_spread)
    T(x·t³)   → НАПРАВЛЕНИЕ (skewness)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Any
from functools import lru_cache


@lru_cache(maxsize=16)
def _get_time_axis(N: int) -> NDArray:
    """Кэшированная временная ось [0, 1]"""
    return np.linspace(0, 1, N)


@lru_cache(maxsize=64)
def _get_t_power(N: int, k: int) -> NDArray:
    """Кэшированная степень временной оси t^k"""
    return _get_time_axis(N) ** k


def _get_transform_func(name: str):
    """Получить функцию преобразования по имени"""
    if name == "fft":
        return np.fft.rfft
    elif name == "dct":
        from scipy.fftpack import dct
        return lambda x: dct(x, type=2, norm="ortho")
    else:
        raise ValueError(f"Unknown transform: {name}")


def compute(
    signal: NDArray,
    transform: str = "fft",
    moments: List[int] = [0, 1, 2],
) -> Dict[str, Any]:
    """
    Вычислить синхронные метрики.

    Args:
        signal: входной сигнал
        transform: "fft" или "dct"
        moments: список моментов [0, 1, 2]

    Returns:
        {
            'magnitude': |T(x)|,
            'time_center': tc[k],
            'time_spread': ts[k],
        }
    """
    signal = np.asarray(signal)
    N = len(signal)

    transform_func = _get_transform_func(transform)

    # T(x · t^k) для каждого k
    transforms = {}
    for k in moments:
        t_power = _get_t_power(N, k)
        weighted = signal * t_power
        transforms[k] = transform_func(weighted)

    # Magnitude
    T0 = transforms[0]
    magnitude = np.abs(T0)
    magnitude_safe = np.maximum(magnitude, 1e-10)

    result = {
        "magnitude": magnitude,
    }

    # Time center: E[t] = |T(x·t)| / |T(x)|
    if 1 in transforms:
        T1 = transforms[1]
        time_center = np.abs(T1) / magnitude_safe
        time_center = np.clip(time_center, 0, 1)
        result["time_center"] = time_center

    # Time spread: sqrt(E[t²] - E[t]²)
    if 2 in transforms and "time_center" in result:
        T2 = transforms[2]
        variance = np.abs(T2) / magnitude_safe - result["time_center"] ** 2
        time_spread = np.sqrt(np.maximum(variance, 0))
        result["time_spread"] = time_spread

    # Skewness
    if 3 in transforms and "time_center" in result and "time_spread" in result:
        T3 = transforms[3]
        tc = result["time_center"]
        ts = result["time_spread"]
        T2 = transforms[2]

        m3 = np.abs(T3) / magnitude_safe - 3 * tc * (np.abs(T2) / magnitude_safe) + 2 * tc**3
        skewness = m3 / (ts**3 + 1e-10)
        result["skewness"] = np.clip(skewness, -10, 10)

    return result


def dominant_metrics(
    signal: NDArray,
    transform: str = "fft",
    moments: List[int] = [0, 1, 2],
) -> Dict[str, float]:
    """
    Метрики для доминирующего коэффициента.

    Returns:
        {'tc': float, 'ts': float, 'dominant_idx': int}
    """
    metrics = compute(signal, transform, moments)

    # Доминирующий коэффициент (максимальная magnitude, пропускаем DC)
    mag = metrics["magnitude"]
    if len(mag) > 1:
        dom_idx = int(np.argmax(mag[1:]) + 1)
    else:
        dom_idx = 0

    result = {"dominant_idx": dom_idx}

    if "time_center" in metrics:
        result["tc"] = float(metrics["time_center"][dom_idx])
    else:
        result["tc"] = 0.5

    if "time_spread" in metrics:
        result["ts"] = float(metrics["time_spread"][dom_idx])
    else:
        result["ts"] = 0.0

    return result


def get_unified_features(
    signal: NDArray,
    transforms: List[str] = ["fft", "dct"],
    moments: List[int] = [0, 1, 2],
) -> Dict[str, float]:
    """
    Унифицированный вектор фичей от нескольких преобразований.

    Returns:
        {'{transform}_{metric}': value}
    """
    features = {}

    for name in transforms:
        if name in ("fft", "dct"):
            m = dominant_metrics(signal, name, moments)
            prefix = name
            features[f"{prefix}_tc"] = m["tc"]
            features[f"{prefix}_ts"] = m["ts"]

        elif name == "hilbert":
            # Envelope через Hilbert
            from scipy.signal import hilbert as scipy_hilbert

            analytic = scipy_hilbert(signal)
            env = np.abs(analytic)
            m = _envelope_metrics(env)
            features["hil_tc"] = m["time_center"]
            features["hil_ts"] = m["time_spread"]

        elif name == "teager":
            # Teager energy operator
            teo = _teager_energy(signal)
            m = _envelope_metrics(teo)
            features["teo_tc"] = m["time_center"]
            features["teo_ts"] = m["time_spread"]

    return features


def _envelope_metrics(envelope: NDArray) -> Dict[str, float]:
    """Метрики для огибающей"""
    envelope = np.asarray(envelope)
    N = len(envelope)
    t = _get_time_axis(N)

    # Нормализуем как распределение
    env_sum = envelope.sum()
    if env_sum < 1e-10:
        return {"time_center": 0.5, "time_spread": 0.0}

    env_norm = envelope / env_sum

    # E[t]
    time_center = float(np.sum(t * env_norm))

    # Var[t]
    variance = float(np.sum((t - time_center) ** 2 * env_norm))
    time_spread = float(np.sqrt(variance))

    return {"time_center": time_center, "time_spread": time_spread}


def _teager_energy(signal: NDArray) -> NDArray:
    """Teager Energy Operator"""
    signal = np.asarray(signal)
    # TEO[n] = x[n]² - x[n-1]·x[n+1]
    x_sq = signal[1:-1] ** 2
    x_prev_next = signal[:-2] * signal[2:]
    teo = x_sq - x_prev_next
    # Pad to original length
    return np.concatenate([[teo[0]], teo, [teo[-1]]])
