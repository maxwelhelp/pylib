"""
БЫСТРЫЙ ТЕСТ - запусти чтобы проверить что всё работает

Использование:
    cd lifting_v2
    python tests/quick_test.py
"""

import sys
import os

# Добавляем родительскую папку в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("="*60)
print("LIFTING v2.0 - QUICK TEST")
print("="*60)

# 1. Импорты
print("\n1. Checking imports...")
try:
    from core import normalize, d_geo, centroid, d_eff
    from transforms import SyncLifting, fft, kmer_frequencies
    from tasks import AnomalyDetector, Classifier
    from domains import ECGAnalyzer, DNAAnalyzer, AudioAnalyzer
    print("   ✅ All imports OK")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# 2. Базовая математика
print("\n2. Core math...")
s = normalize([3, 4, 0])
assert abs(np.linalg.norm(s) - 1.0) < 1e-10
print(f"   normalize([3,4,0]) = {s} ✅")

d = d_geo(normalize([1,0]), normalize([0,1]))
assert abs(np.degrees(d) - 90) < 0.1
print(f"   d_geo = {np.degrees(d):.0f}° ✅")

# 3. Sync Lifting
print("\n3. Sync Lifting...")
signal = np.sin(np.linspace(0, 4*np.pi, 256))
sync = SyncLifting(moments=[0, 1, 2])
m = sync.dominant_metrics(signal, np.fft.rfft)
print(f"   tc={m['tc']:.3f}, ts={m['ts']:.3f} ✅")

# 4. DNA
print("\n4. DNA Analyzer...")
dna = DNAAnalyzer(k=2)
features = dna.extract_features("ATGCGATCGATC")
print(f"   Extracted {len(features)} features ✅")

# 5. Anomaly Detector
print("\n5. Anomaly Detector...")
train = np.array([normalize(np.random.randn(10) + [1,0,0,0,0,0,0,0,0,0]) for _ in range(20)])
detector = AnomalyDetector(threshold_sigma=2.0)
detector.fit(train)
print(f"   Threshold: {np.degrees(detector.threshold):.1f}° ✅")

print("\n" + "="*60)
print("ALL TESTS PASSED! ✅")
print("="*60)
print("\nБиблиотека работает. Можно использовать.")
