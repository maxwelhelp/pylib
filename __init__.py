"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         LIFTING LIBRARY v2.0                                 ║
║                                                                              ║
║  Универсальная геометрическая библиотека для представления данных            ║
║                                                                              ║
║  Архитектура:                                                                ║
║    core/       - Ядро: геометрия, shape, метрики (НЕ ТРОГАТЬ)               ║
║    transforms/ - Кубики: FFT, wavelet, sync lifting, k-mer                   ║
║    tasks/      - Задачи: anomaly, classifier                                 ║
║    domains/    - Примеры: ECG, DNA, Audio (копируй и изменяй)               ║
║                                                                              ║
║  Уровни использования:                                                       ║
║    Новичок    → domains.ECGAnalyzer()                                        ║
║    Практик    → transforms + tasks → свой domain                             ║
║    Исследователь → core + новые transforms                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = "2.0.0"
__author__ = "Lifting Theory Research"

# ============================================================
# CORE (ядро - универсальная математика)
# ============================================================

from .core import (
    # Geometry
    Sphere,
    d_geo,
    d_geo_deg,
    centroid,
    slerp,
    frechet_mean,
    distance_matrix,
    
    # Shape
    normalize,
    normalize_with_energy,
    Shape,
    to_shape,
    dict_to_shape,
    concat_and_normalize,
    
    # Metrics
    d_eff,
    d_eff_spectrum,
    kl_divergence,
    separation_ratio,
)

# ============================================================
# TRANSFORMS (кубики - преобразования)
# ============================================================

from .transforms import (
    # Spectral
    fft,
    dct,
    stft,
    
    # Wavelet
    cwt,
    dwt,
    wavelet_energy,
    
    # Analytic
    hilbert,
    envelope,
    teager_energy,
    autocorrelation,
    cepstrum,
    
    # Sync Lifting (ключевой!)
    SyncLifting,
    sync_fft,
    get_unified_features,
    
    # Sequence
    kmer_frequencies,
    gc_content,
    
    # Registry
    get_transform,
    register_transform,
    list_transforms,
)

# ============================================================
# TASKS (задачи)
# ============================================================

from .tasks import (
    AnomalyDetector,
    AnomalyResult,
    detect_anomalies,
    
    Classifier,
    ClassificationResult,
    classify,
)

# ============================================================
# DOMAINS (готовые анализаторы)
# ============================================================

from .domains import (
    BaseDomainAnalyzer,
    ECGAnalyzer,
    DNAAnalyzer,
    AudioAnalyzer,
    TTSAnalyzer,
    TemplateDomainAnalyzer,
)


# ============================================================
# QUICK START FUNCTIONS
# ============================================================

def quick_shape(data):
    """
    Быстрое преобразование данных → shape.
    
    Для любых данных (сигнал, вектор, словарь).
    
    Examples:
        >>> s = quick_shape([1, 2, 3, 4])
        >>> np.linalg.norm(s)  # 1.0
    """
    import numpy as np
    
    if isinstance(data, dict):
        return dict_to_shape(data)
    else:
        return normalize(np.asarray(data).flatten())


def quick_distance(x, y):
    """
    Быстрое расстояние между двумя объектами.
    
    Returns:
        Угол в градусах
    """
    s1 = quick_shape(x)
    s2 = quick_shape(y)
    return d_geo_deg(s1, s2)


def quick_analyze(signals, transforms=['fft'], moments=[0, 1, 2]):
    """
    Быстрый анализ набора сигналов.
    
    Returns:
        {'shapes': array, 'd_eff': int, 'centroid': array}
    """
    import numpy as np
    
    shapes = []
    for sig in signals:
        features = get_unified_features(sig, transforms, moments)
        shapes.append(dict_to_shape(features))
    
    shapes = np.array(shapes)
    
    return {
        'shapes': shapes,
        'd_eff': d_eff(shapes),
        'centroid': centroid(shapes),
    }


__all__ = [
    # Version
    '__version__',
    
    # Core
    'Sphere', 'd_geo', 'd_geo_deg', 'centroid', 'slerp', 'frechet_mean',
    'normalize', 'Shape', 'to_shape', 'dict_to_shape',
    'd_eff', 'kl_divergence', 'separation_ratio',
    
    # Transforms
    'fft', 'dct', 'stft', 'cwt', 'dwt',
    'hilbert', 'envelope', 'teager_energy', 'autocorrelation', 'cepstrum',
    'SyncLifting', 'sync_fft', 'get_unified_features',
    'kmer_frequencies', 'gc_content',
    'get_transform', 'register_transform', 'list_transforms',
    
    # Tasks
    'AnomalyDetector', 'detect_anomalies',
    'Classifier', 'classify',
    
    # Domains
    'BaseDomainAnalyzer', 'ECGAnalyzer', 'DNAAnalyzer', 
    'AudioAnalyzer', 'TTSAnalyzer', 'TemplateDomainAnalyzer',
    
    # Quick functions
    'quick_shape', 'quick_distance', 'quick_analyze',
]
