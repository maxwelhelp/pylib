"""
Audio Domain Analyzer
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any
from .base import BaseDomainAnalyzer

# Универсальные импорты
try:
    from ..transforms import SyncLifting
    from ..core import d_eff
except ImportError:
    from transforms import SyncLifting
    from core import d_eff


class AudioAnalyzer(BaseDomainAnalyzer):
    """
    Анализатор аудио сигналов.
    
    Examples:
        >>> analyzer = AudioAnalyzer(fs=16000)
        >>> shapes = analyzer.process(audio)
    """
    
    default_config = {
        'fs': 16000,
        'frame_ms': 25,
        'hop_ms': 10,
        'features': ['fft', 'energy'],
        'n_fft_bins': 64,
    }
    
    def preprocess(self, data: Any) -> List[NDArray]:
        """Audio → frames"""
        signal = np.asarray(data, dtype=np.float64)
        
        if np.max(np.abs(signal)) > 1.0:
            signal = signal / 32768.0
        
        fs = self.config['fs']
        frame_size = int(self.config['frame_ms'] * fs / 1000)
        hop_size = int(self.config['hop_ms'] * fs / 1000)
        
        frames = []
        for i in range(0, len(signal) - frame_size, hop_size):
            frames.append(signal[i:i + frame_size])
        return frames
    
    def extract_features(self, frame: NDArray) -> Dict[str, float]:
        """Frame → features"""
        features = {}
        windowed = frame * np.hanning(len(frame))
        
        for feat_name in self.config['features']:
            if feat_name == 'fft':
                sync = SyncLifting(moments=[0, 1, 2])
                m = sync.dominant_metrics(windowed, np.fft.rfft)
                features['fft_tc'] = m.get('tc', 0.5)
                features['fft_ts'] = m.get('ts', 0.0)
                
            elif feat_name == 'energy':
                features['energy'] = float(np.sum(frame ** 2))
        
        return features
    
    def compute_d_eff(self, signal: NDArray) -> int:
        """d_eff для сигнала"""
        shapes = self.process(signal)
        return d_eff(shapes)


class TTSAnalyzer(AudioAnalyzer):
    """Анализатор для TTS"""
    
    default_config = {
        'fs': 22050,
        'frame_ms': 25,
        'hop_ms': 10,
        'features': ['fft'],
    }
