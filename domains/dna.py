"""
DNA Domain Analyzer
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any
from .base import BaseDomainAnalyzer

# Универсальные импорты
try:
    from ..transforms import kmer_frequencies, gc_content, sequence_complexity
    from ..core import d_eff  # d_eff локальная (не секретная)
    from ..tasks.backend import get_backend
except ImportError:
    from transforms import kmer_frequencies, gc_content, sequence_complexity
    from core import d_eff
    from tasks.backend import get_backend


class DNAAnalyzer(BaseDomainAnalyzer):
    """
    Анализатор ДНК последовательностей.
    
    Examples:
        >>> analyzer = DNAAnalyzer(k=4)
        >>> analyzer.fit_classifier(sequences, labels)
        >>> result = analyzer.classify(new_sequence)
    """
    
    default_config = {
        'k': 4,
        'window_size': 1000,
        'use_gc': True,
        'use_complexity': True,
    }
    
    def preprocess(self, data: Any) -> List[str]:
        """DNA → windows"""
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
        
        sequence = str(data).upper()
        sequence = ''.join(c for c in sequence if c in 'ACGT')
        
        window_size = self.config['window_size']
        if len(sequence) <= window_size:
            return [sequence]
        
        windows = []
        for i in range(0, len(sequence) - window_size + 1, window_size):
            windows.append(sequence[i:i + window_size])
        return windows
    
    def extract_features(self, segment: str) -> Dict[str, float]:
        """DNA window → features"""
        features = {}
        
        kmer_freq = kmer_frequencies(segment, self.config['k'])
        for i, freq in enumerate(kmer_freq):
            features[f'kmer_{i}'] = float(freq)
        
        if self.config['use_gc']:
            features['gc_content'] = gc_content(segment)
        
        if self.config['use_complexity']:
            features['complexity'] = sequence_complexity(segment, k=3)
        
        return features
    
    def analyze_sequence(self, sequence: str) -> Dict[str, Any]:
        """Полный анализ последовательности"""
        sequence = str(sequence).upper()
        sequence = ''.join(c for c in sequence if c in 'ACGT')
        shapes = self.process(sequence)
        
        return {
            'length': len(sequence),
            'gc_content': gc_content(sequence),
            'complexity': sequence_complexity(sequence),
            'n_windows': len(shapes),
            'd_eff': d_eff(shapes) if len(shapes) > 1 else shapes.shape[-1],
        }
    
    def compare_sequences(self, seq1: str, seq2: str) -> Dict[str, float]:
        """Сравнение двух последовательностей"""
        shapes1 = self.process(seq1)
        shapes2 = self.process(seq2)

        backend = get_backend()
        c1 = backend.centroid(shapes1) if len(shapes1) > 1 else shapes1[0]
        c2 = backend.centroid(shapes2) if len(shapes2) > 1 else shapes2[0]

        d = backend.d_geo(c1, c2)
        return {'distance': float(d), 'distance_deg': float(np.degrees(d))}
