"""
ECG Domain Analyzer
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseDomainAnalyzer

# Универсальные импорты
try:
    from ..transforms import SyncLifting, envelope, teager_energy
    from ..tasks import AnomalyDetector
except ImportError:
    from transforms import SyncLifting, envelope, teager_energy
    from tasks import AnomalyDetector


class ECGAnalyzer(BaseDomainAnalyzer):
    """
    Анализатор ЭКГ сигналов.
    
    Examples:
        >>> analyzer = ECGAnalyzer()
        >>> analyzer.fit_anomaly(ecg, annotations)
        >>> results = analyzer.detect_anomalies(ecg)
    """
    
    default_config = {
        'fs': 360,
        'window_ms': 400,
        'transforms': ['fft', 'hilbert', 'teager'],
        'threshold_sigma': 1.5,
    }
    
    def __init__(self, **config):
        super().__init__(**config)
        self._annotations = None
        self._labels = None
    
    def preprocess(self, data: Any) -> List[NDArray]:
        """ECG → beats"""
        if isinstance(data, tuple):
            signal, annotations = data
            self._annotations = annotations
        else:
            signal = data
            annotations = self._annotations
        
        signal = np.asarray(signal)
        
        if annotations is not None:
            return self._extract_from_annotations(signal, annotations)
        else:
            return self._detect_beats(signal)
    
    def _extract_from_annotations(self, signal: NDArray, annotations: Any) -> List[NDArray]:
        """Извлечение beats по аннотациям"""
        fs = self.config['fs']
        window_samples = int(self.config['window_ms'] * fs / 1000)
        half = window_samples // 2
        
        # Парсим аннотации
        if hasattr(annotations, 'sample'):
            samples = annotations.sample
            symbols = annotations.symbol
        elif isinstance(annotations, dict):
            samples = annotations.get('samples', annotations.get('sample', []))
            symbols = annotations.get('symbols', annotations.get('symbol', []))
        else:
            samples = annotations
            symbols = ['N'] * len(samples)
        
        beats = []
        labels = []
        
        for sample, symbol in zip(samples, symbols):
            if symbol in ['+', '~', '|', '[', ']', '!', '"']:
                continue
            if sample < half or sample > len(signal) - half:
                continue
            
            beat = signal[sample - half : sample + half]
            if len(beat) == window_samples:
                beat = (beat - beat.mean()) / (beat.std() + 1e-10)
                beats.append(beat)
                labels.append(str(symbol))
        
        self._labels = labels
        return beats
    
    def _detect_beats(self, signal: NDArray) -> List[NDArray]:
        """Автодетекция R-пиков"""
        from scipy.signal import find_peaks
        
        fs = self.config['fs']
        window_samples = int(self.config['window_ms'] * fs / 1000)
        half = window_samples // 2
        
        peaks, _ = find_peaks(signal, distance=int(0.3*fs), height=np.std(signal))
        
        beats = []
        for peak in peaks:
            if peak < half or peak > len(signal) - half:
                continue
            beat = signal[peak - half : peak + half]
            if len(beat) == window_samples:
                beat = (beat - beat.mean()) / (beat.std() + 1e-10)
                beats.append(beat)
        
        self._labels = ['N'] * len(beats)
        return beats
    
    def extract_features(self, beat: NDArray) -> Dict[str, float]:
        """Beat → features"""
        from scipy.fftpack import dct
        
        features = {}
        sync = SyncLifting(moments=[0, 1, 2])
        
        for name in self.config['transforms']:
            if name == 'fft':
                m = sync.compute(beat, np.fft.rfft)
                dom = np.argmax(m['magnitude'][1:]) + 1
                features['fft_tc'] = float(m['time_center'][dom])
                features['fft_ts'] = float(m['time_spread'][dom])
                
            elif name == 'dct':
                m = sync.compute(beat, lambda x: dct(x, type=2, norm='ortho'))
                dom = np.argmax(np.abs(m['magnitude'][1:])) + 1
                features['dct_tc'] = float(m['time_center'][dom])
                
            elif name == 'hilbert':
                env = envelope(beat)
                m = sync.compute_envelope_metrics(env)
                features['hil_tc'] = m['time_center']
                features['hil_ts'] = m['time_spread']
                
            elif name == 'teager':
                teo = teager_energy(beat)
                m = sync.compute_envelope_metrics(teo)
                features['teo_tc'] = m['time_center']
        
        return features
    
    def get_labels(self) -> List[str]:
        return self._labels if self._labels else []
