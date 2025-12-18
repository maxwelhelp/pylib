"""
Analytic Transforms - Hilbert, Teager, мгновенные характеристики

Преобразования для анализа огибающей, фазы, частоты.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional


def hilbert(signal: NDArray) -> NDArray:
    """
    Hilbert Transform → Аналитический сигнал
    
    z(t) = x(t) + j·H[x](t)
    
    Returns:
        Комплексный аналитический сигнал
    """
    from scipy.signal import hilbert as scipy_hilbert
    return scipy_hilbert(signal)


def envelope(signal: NDArray) -> NDArray:
    """
    Огибающая (мгновенная амплитуда)
    
    A(t) = |z(t)| = |x(t) + j·H[x](t)|
    """
    return np.abs(hilbert(signal))


def instant_phase(signal: NDArray, unwrap: bool = True) -> NDArray:
    """
    Мгновенная фаза
    
    φ(t) = arg(z(t))
    
    Args:
        signal: входной сигнал
        unwrap: развернуть фазу (убрать скачки)
    """
    phase = np.angle(hilbert(signal))
    if unwrap:
        phase = np.unwrap(phase)
    return phase


def instant_frequency(signal: NDArray, fs: float = 1.0) -> NDArray:
    """
    Мгновенная частота
    
    f(t) = (1/2π) · dφ/dt
    
    Args:
        signal: входной сигнал
        fs: частота дискретизации
    """
    phase = instant_phase(signal, unwrap=True)
    return np.gradient(phase) * fs / (2 * np.pi)


def analytic_decomposition(signal: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Полное аналитическое разложение
    
    Returns:
        (amplitude, phase, frequency)
    """
    z = hilbert(signal)
    
    amplitude = np.abs(z)
    phase = np.unwrap(np.angle(z))
    frequency = np.gradient(phase) / (2 * np.pi)
    
    return amplitude, phase, frequency


def teager_energy(signal: NDArray) -> NDArray:
    """
    Teager Energy Operator (TEO)
    
    TEO[n] = x[n]² - x[n-1]·x[n+1]
    
    Мгновенная энергия. Хорошо выделяет события.
    """
    teo = signal[1:-1] ** 2 - signal[:-2] * signal[2:]
    teo = np.abs(teo)
    
    # Дополняем до исходной длины
    return np.concatenate([[teo[0]], teo, [teo[-1]]])


def zero_crossing_rate(signal: NDArray) -> float:
    """
    Частота пересечения нуля
    
    ZCR = (1/N) · Σ |sign(x[n]) - sign(x[n-1])|
    """
    signs = np.sign(signal)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / len(signal))


def zero_crossing_times(signal: NDArray) -> NDArray:
    """
    Времена пересечения нуля
    
    Returns:
        Индексы точек пересечения
    """
    return np.where(np.diff(np.sign(signal)))[0]


def autocorrelation(signal: NDArray, n_coeffs: Optional[int] = None,
                    normalized: bool = True) -> NDArray:
    """
    Автокорреляция
    
    r(τ) = Σ x(t)·x(t+τ)
    
    Shift-invariant! Хорошо для периодических сигналов.
    """
    from scipy.signal import correlate
    
    result = correlate(signal, signal, mode='full')
    center = len(result) // 2
    result = result[center:]  # только положительные лаги
    
    if normalized and result[0] != 0:
        result = result / result[0]
    
    if n_coeffs is not None:
        result = result[:n_coeffs]
    
    return result


def cepstrum(signal: NDArray, n_coeffs: Optional[int] = None) -> NDArray:
    """
    Real Cepstrum
    
    c(t) = IFFT(log|FFT(x)|)
    
    Разделяет свёрточные компоненты (source/filter).
    Хорошо для речи.
    """
    spectrum = np.fft.fft(signal)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    ceps = np.real(np.fft.ifft(log_spectrum))
    
    if n_coeffs is not None:
        ceps = ceps[:n_coeffs]
    
    return ceps



