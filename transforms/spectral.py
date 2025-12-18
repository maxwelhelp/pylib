"""
Spectral Transforms - FFT, DCT, STFT

Базовые спектральные преобразования.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional


def fft(signal: NDArray, n_coeffs: Optional[int] = None) -> NDArray:
    """
    Fast Fourier Transform (magnitude)
    
    Args:
        signal: входной сигнал
        n_coeffs: количество коэффициентов (по умолчанию все)
        
    Returns:
        Амплитудный спектр (положительные частоты)
    """
    spectrum = np.fft.rfft(signal)
    magnitude = np.abs(spectrum)
    
    if n_coeffs is not None:
        magnitude = magnitude[:n_coeffs]
    
    return magnitude


def fft_complex(signal: NDArray) -> NDArray:
    """FFT с комплексным результатом"""
    return np.fft.rfft(signal)


def ifft(spectrum: NDArray, n: Optional[int] = None) -> NDArray:
    """Inverse FFT"""
    return np.real(np.fft.irfft(spectrum, n=n))


def dct(signal: NDArray, n_coeffs: Optional[int] = None) -> NDArray:
    """
    Discrete Cosine Transform (Type II)
    
    Используется в JPEG, MP3. Хорошо сжимает.
    """
    from scipy.fftpack import dct as scipy_dct
    
    coeffs = scipy_dct(signal, type=2, norm='ortho')
    
    if n_coeffs is not None:
        coeffs = coeffs[:n_coeffs]
    
    return coeffs


def dst(signal: NDArray, n_coeffs: Optional[int] = None) -> NDArray:
    """Discrete Sine Transform (Type II)"""
    from scipy.fftpack import dst as scipy_dst
    
    coeffs = scipy_dst(signal, type=2, norm='ortho')
    
    if n_coeffs is not None:
        coeffs = coeffs[:n_coeffs]
    
    return coeffs


def stft(signal: NDArray, 
         window_size: int = 256,
         hop_size: Optional[int] = None,
         window: str = 'hann') -> NDArray:
    """
    Short-Time Fourier Transform
    
    Args:
        signal: входной сигнал
        window_size: размер окна
        hop_size: шаг (по умолчанию window_size // 4)
        window: тип окна ('hann', 'hamming', 'rect')
        
    Returns:
        [n_frames, n_freq] спектрограмма (magnitude)
    """
    if hop_size is None:
        hop_size = window_size // 4
    
    # Окно
    if window == 'hann':
        win = np.hanning(window_size)
    elif window == 'hamming':
        win = np.hamming(window_size)
    else:
        win = np.ones(window_size)
    
    n_frames = (len(signal) - window_size) // hop_size + 1
    
    frames = []
    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start:start + window_size] * win
        spectrum = np.abs(np.fft.rfft(frame))
        frames.append(spectrum)
    
    return np.array(frames)


def power_spectrum(signal: NDArray) -> NDArray:
    """Power spectrum |FFT|²"""
    return np.abs(np.fft.rfft(signal)) ** 2


def log_power_spectrum(signal: NDArray, eps: float = 1e-10) -> NDArray:
    """Log power spectrum (в dB)"""
    return np.log(power_spectrum(signal) + eps)


def spectral_centroid(signal: NDArray) -> float:
    """
    Спектральный центроид — "яркость" звука
    
    centroid = Σ(f × |X(f)|) / Σ|X(f)|
    """
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.arange(len(spectrum))
    
    return float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10))


def spectral_bandwidth(signal: NDArray) -> float:
    """Спектральная ширина"""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.arange(len(spectrum))
    
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / (np.sum(spectrum) + 1e-10))
    
    return float(bandwidth)
