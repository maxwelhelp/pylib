"""
ТЕСТ НА РЕАЛЬНЫХ ECG ДАННЫХ (MIT-BIH Record 200)

Использование:
    cd lifting_v2
    pip install wfdb
    python tests/test_real_ecg.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import Counter

print("="*60)
print("REAL ECG TEST - MIT-BIH Record 200")
print("="*60)

# Проверяем наличие данных
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
data_file = os.path.join(data_dir, '200')

if not os.path.exists(data_file + '.dat'):
    print(f"\n❌ Данные не найдены: {data_file}.dat")
    print("   Скачайте MIT-BIH Record 200 и положите в папку data/")
    sys.exit(1)

try:
    import wfdb
except ImportError:
    print("\n❌ wfdb не установлен")
    print("   pip install wfdb")
    sys.exit(1)

from core import normalize, d_eff
from transforms import SyncLifting, envelope, teager_energy
from tasks import AnomalyDetector
from scipy.fftpack import dct

# 1. Загрузка данных
print("\n1. Loading data...")
record = wfdb.rdrecord(data_file)
annotation = wfdb.rdann(data_file, 'atr')

ecg = record.p_signal[:, 0]
fs = record.fs

print(f"   Duration: {len(ecg)/fs:.0f} sec")
print(f"   Sample rate: {fs} Hz")
print(f"   Annotations: {len(annotation.sample)}")

# 2. Извлечение ударов
print("\n2. Extracting beats...")
window_ms = 400
window_samples = int(window_ms * fs / 1000)
half = window_samples // 2

beats = []
labels = []

for sample, symbol in zip(annotation.sample, annotation.symbol):
    if symbol in ['+', '~', '|', '[', ']', '!', '"']:
        continue
    if sample < half or sample > len(ecg) - half:
        continue
    
    beat = ecg[sample - half : sample + half]
    if len(beat) == window_samples:
        beat = (beat - beat.mean()) / (beat.std() + 1e-10)
        beats.append(beat)
        labels.append(symbol)

print(f"   Extracted: {len(beats)} beats")
print(f"   Labels: {dict(Counter(labels))}")

# 3. Feature extraction
print("\n3. Building shapes with SyncLifting...")

def extract_features(beat):
    sync = SyncLifting(moments=[0, 1, 2])
    features = {}
    
    # FFT
    m = sync.compute(beat, np.fft.rfft)
    dom = np.argmax(m['magnitude'][1:]) + 1
    features['fft_tc'] = float(m['time_center'][dom])
    features['fft_ts'] = float(m['time_spread'][dom])
    
    # DCT
    m = sync.compute(beat, lambda x: dct(x, type=2, norm='ortho'))
    dom = np.argmax(np.abs(m['magnitude'][1:])) + 1
    features['dct_tc'] = float(m['time_center'][dom])
    
    # Hilbert
    env = envelope(beat)
    env_m = sync.compute_envelope_metrics(env)
    features['hil_tc'] = env_m['time_center']
    features['hil_ts'] = env_m['time_spread']
    
    # Teager
    teo = teager_energy(beat)
    teo_m = sync.compute_envelope_metrics(teo)
    features['teo_tc'] = teo_m['time_center']
    
    return features

shapes = np.array([normalize(np.array(list(extract_features(b).values()))) for b in beats])
labels_arr = np.array(labels)

print(f"   Shape dim: {shapes.shape[-1]}")
print(f"   d_eff: {d_eff(shapes)}")

# 4. Training
print("\n4. Training on normal beats...")
normal_idx = np.where(labels_arr == 'N')[0]
train_idx = normal_idx[:500]

detector = AnomalyDetector(threshold_sigma=1.5)
detector.fit(shapes[train_idx])
print(f"   Train size: {len(train_idx)}")
print(f"   Threshold: {np.degrees(detector.threshold):.1f}°")

# 5. Testing
print("\n5. Detection results:")
results = detector.predict(shapes)

for label in ['N', 'V', 'A', 'F']:
    idx = np.where(labels_arr == label)[0]
    if len(idx) == 0:
        continue
    detected = sum(1 for i in idx if results[i].is_anomaly)
    print(f"   {label}: {detected}/{len(idx)} ({100*detected/len(idx):.0f}%)")

# 6. Final metrics
is_anomaly_true = labels_arr != 'N'
is_anomaly_pred = np.array([r.is_anomaly for r in results])

tp = np.sum(is_anomaly_pred & is_anomaly_true)
tn = np.sum(~is_anomaly_pred & ~is_anomaly_true)
fp = np.sum(is_anomaly_pred & ~is_anomaly_true)
fn = np.sum(~is_anomaly_pred & is_anomaly_true)

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"   Sensitivity: {sensitivity:.0%} (из аритмий сколько нашли)")
print(f"   Specificity: {specificity:.0%} (нормальных не тронули)")
print("="*60)
