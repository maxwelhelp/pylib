"""
Template Domain Analyzer - ШАБЛОН для своего домена

Инструкция:
1. Скопируй этот файл
2. Переименуй класс
3. Задай default_config
4. Реализуй preprocess() и extract_features()
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any
from .base import BaseDomainAnalyzer

# Универсальные импорты
try:
    from ..transforms import SyncLifting
    from ..tasks.backend import get_backend
except ImportError:
    from transforms import SyncLifting
    from tasks.backend import get_backend


class TemplateDomainAnalyzer(BaseDomainAnalyzer):
    """
    Шаблон анализатора. Скопируй и измени.
    """
    
    default_config = {
        'window_size': 256,
        'transforms': ['fft'],
        'use_sync_lifting': True,
        'moments': [0, 1, 2],
    }
    
    def preprocess(self, data: Any) -> List[Any]:
        """Данные → сегменты. ПЕРЕОПРЕДЕЛИ."""
        data = np.asarray(data)
        window_size = self.config['window_size']
        
        segments = []
        for i in range(0, len(data) - window_size + 1, window_size):
            segments.append(data[i:i + window_size])
        return segments
    
    def extract_features(self, segment: Any) -> Dict[str, float]:
        """Сегмент → фичи. ПЕРЕОПРЕДЕЛИ."""
        features = {}
        segment = np.asarray(segment)
        
        if self.config['use_sync_lifting']:
            sync = SyncLifting(moments=self.config['moments'])
            m = sync.dominant_metrics(segment, np.fft.rfft)
            features['fft_tc'] = m.get('tc', 0.5)
            features['fft_ts'] = m.get('ts', 0.0)
        else:
            features['mean'] = float(np.mean(segment))
            features['std'] = float(np.std(segment))
        
        return features


# ============================================================
# ПРИМЕР: Сейсмический анализатор
# ============================================================

class SeismicAnalyzer(BaseDomainAnalyzer):
    """Пример кастомного домена"""
    
    default_config = {
        'window_size': 512,
        'fs': 100,
    }
    
    def preprocess(self, data):
        data = np.asarray(data)
        window_size = self.config['window_size']
        step = window_size // 2
        
        segments = []
        for i in range(0, len(data) - window_size, step):
            segments.append(data[i:i + window_size])
        return segments
    
    def extract_features(self, segment):
        sync = SyncLifting(moments=[0, 1, 2])
        m = sync.dominant_metrics(segment, np.fft.rfft)
        
        # STA/LTA
        sta_len = len(segment) // 10
        sta = np.mean(np.abs(segment[-sta_len:]))
        lta = np.mean(np.abs(segment))
        
        return {
            'fft_tc': m.get('tc', 0.5),
            'fft_ts': m.get('ts', 0.0),
            'sta_lta': float(sta / (lta + 1e-10)),
        }
