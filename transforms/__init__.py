"""
Transforms - Кубики преобразований

Все преобразования сигналов и последовательностей.
Комбинируй как хочешь для создания своего домена.

Категории:
- spectral: fft, dct, stft
- wavelet: cwt, dwt
- analytic: hilbert, teager, envelope
- sync: SyncLifting (ключевой!)
- sequence: kmer, ngram (для DNA, text)
"""

# Spectral
from .spectral import (
    fft,
    fft_complex,
    ifft,
    dct,
    dst,
    stft,
    power_spectrum,
    log_power_spectrum,
    spectral_centroid,
    spectral_bandwidth,
)

# Wavelet
from .wavelet import (
    cwt,
    dwt,
    wavelet_energy,
    wavelet_features,
)

# Analytic
from .analytic import (
    hilbert,
    envelope,
    instant_phase,
    instant_frequency,
    analytic_decomposition,
    teager_energy,
    zero_crossing_rate,
    zero_crossing_times,
    autocorrelation,
    cepstrum,
)

# Sync Lifting (ключевой!)
from .sync import (
    SyncLifting,
    sync_fft,
    sync_dct,
    get_unified_features,
)

# Sequence
from .sequence import (
    kmer_frequencies,
    kmer_spectrum,
    gc_content,
    dna_to_numeric,
    one_hot_encode,
    ngram_frequencies,
    transition_matrix,
    sequence_complexity,
)


__all__ = [
    # Spectral
    'fft', 'fft_complex', 'ifft', 'dct', 'dst', 'stft',
    'power_spectrum', 'log_power_spectrum',
    'spectral_centroid', 'spectral_bandwidth',
    
    # Wavelet
    'cwt', 'dwt', 'wavelet_energy', 'wavelet_features',
    
    # Analytic
    'hilbert', 'envelope', 'instant_phase', 'instant_frequency',
    'analytic_decomposition', 'teager_energy',
    'zero_crossing_rate', 'zero_crossing_times',
    'autocorrelation', 'cepstrum',
    
    # Sync Lifting
    'SyncLifting', 'sync_fft', 'sync_dct', 'get_unified_features',
    
    # Sequence
    'kmer_frequencies', 'kmer_spectrum', 'gc_content',
    'dna_to_numeric', 'one_hot_encode',
    'ngram_frequencies', 'transition_matrix', 'sequence_complexity',
]


# ============================================================
# REGISTRY (для динамического доступа)
# ============================================================

_TRANSFORM_REGISTRY = {
    'fft': fft,
    'dct': dct,
    'dst': dst,
    'stft': stft,
    'cwt': cwt,
    'dwt': dwt,
    'hilbert': hilbert,
    'envelope': envelope,
    'teager': teager_energy,
    'autocorr': autocorrelation,
    'cepstrum': cepstrum,
    'kmer': kmer_frequencies,
}


def get_transform(name: str):
    """
    Получить transform по имени
    
    Examples:
        >>> transform = get_transform('fft')
        >>> result = transform(signal)
    """
    if name in _TRANSFORM_REGISTRY:
        return _TRANSFORM_REGISTRY[name]
    raise ValueError(f"Unknown transform: {name}. Available: {list(_TRANSFORM_REGISTRY.keys())}")


def register_transform(name: str, func):
    """
    Зарегистрировать новый transform
    
    Для исследователей: добавить свой transform в систему.
    
    Examples:
        >>> def my_transform(signal):
        ...     return custom_processing(signal)
        >>> register_transform('my_transform', my_transform)
    """
    _TRANSFORM_REGISTRY[name] = func


def list_transforms():
    """Список доступных transforms"""
    return list(_TRANSFORM_REGISTRY.keys())
