"""
ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ LIFTING v2.0

Запуск:
    cd lifting_v2
    python tests/examples.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("="*60)
print("LIFTING v2.0 - ПРИМЕРЫ")
print("="*60)


# ============================================================
# ПРИМЕР 1: НОВИЧОК - DNA классификация
# ============================================================

print("\n" + "="*60)
print("ПРИМЕР 1: DNA классификация (Новичок)")
print("="*60)

from domains import DNAAnalyzer

# Данные
gc_sequences = ["GCGCGCGCGCGCGCGC" * 5 for _ in range(10)]
at_sequences = ["ATATATATATATATAT" * 5 for _ in range(10)]

# Обучение - 3 строки!
analyzer = DNAAnalyzer(k=2)
analyzer.fit_classifier(gc_sequences + at_sequences, ['GC']*10 + ['AT']*10)

# Предсказание
test = "GCGCGCGCGCGCGCGCGCGCGC"
result = analyzer.classify([test])[0]
print(f"Последовательность: {test[:20]}...")
print(f"Предсказано: {result['predicted']}")


# ============================================================
# ПРИМЕР 2: ПРАКТИК - Свой детектор аномалий
# ============================================================

print("\n" + "="*60)
print("ПРИМЕР 2: Детектор аномалий (Практик)")
print("="*60)

from transforms import SyncLifting
from tasks import AnomalyDetector
from core import normalize

# Свой анализатор
class MySignalAnalyzer:
    def __init__(self):
        self.sync = SyncLifting(moments=[0, 1, 2])
        self.detector = AnomalyDetector(threshold_sigma=2.0)
    
    def get_shape(self, signal):
        m = self.sync.dominant_metrics(signal, np.fft.rfft)
        return normalize([m['tc'], m['ts']])
    
    def fit(self, signals):
        shapes = np.array([self.get_shape(s) for s in signals])
        self.detector.fit(shapes)
    
    def check(self, signal):
        shape = self.get_shape(signal)
        result = self.detector.predict([shape])[0]
        return result.is_anomaly, result.distance_deg

# Генерируем данные
t = np.linspace(0, 2*np.pi, 256)
normal = [np.sin(5*t + np.random.randn()*0.1) for _ in range(20)]
anomaly = np.sin(20*t)  # Другая частота

# Обучаем и проверяем
my_analyzer = MySignalAnalyzer()
my_analyzer.fit(normal)

is_anom, dist = my_analyzer.check(normal[0])
print(f"Нормальный сигнал: аномалия={is_anom}, расстояние={dist:.1f}°")

is_anom, dist = my_analyzer.check(anomaly)
print(f"Аномальный сигнал: аномалия={is_anom}, расстояние={dist:.1f}°")


# ============================================================
# ПРИМЕР 3: Sync Lifting - видим ГДЕ событие
# ============================================================

print("\n" + "="*60)
print("ПРИМЕР 3: Sync Lifting vs обычный FFT")
print("="*60)

from transforms import SyncLifting

t = np.linspace(0, 1, 256)

# Два импульса - в начале и в конце
pulse_start = np.exp(-((t - 0.2) / 0.05)**2)
pulse_end = np.exp(-((t - 0.8) / 0.05)**2)

# Обычный FFT - не различает
fft_start = np.abs(np.fft.rfft(pulse_start))
fft_end = np.abs(np.fft.rfft(pulse_end))
fft_diff = np.linalg.norm(fft_start - fft_end)

# Sync Lifting - различает!
sync = SyncLifting(moments=[0, 1, 2])
m_start = sync.dominant_metrics(pulse_start, np.fft.rfft)
m_end = sync.dominant_metrics(pulse_end, np.fft.rfft)

print(f"Импульс в начале (t=0.2):")
print(f"  Обычный FFT: некое число")
print(f"  Sync tc = {m_start['tc']:.2f} (показывает позицию!)")

print(f"\nИмпульс в конце (t=0.8):")
print(f"  Обычный FFT: то же самое (разница {fft_diff:.4f})")
print(f"  Sync tc = {m_end['tc']:.2f} (видит разницу!)")


# ============================================================
# ПРИМЕР 4: Сжатие данных (d_eff)
# ============================================================

print("\n" + "="*60)
print("ПРИМЕР 4: Эффективная размерность")
print("="*60)

from core import d_eff, normalize

# Данные лежат на линии (1D структура в 10D пространстве)
line_data = np.array([normalize([1, t, 0, 0, 0, 0, 0, 0, 0, 0]) 
                      for t in np.linspace(-1, 1, 50)])

# Случайные данные (используют всё пространство)
random_data = np.array([normalize(np.random.randn(10)) for _ in range(50)])

print(f"Линия в 10D: d_eff = {d_eff(line_data)} (должно быть ~1)")
print(f"Случайные в 10D: d_eff = {d_eff(random_data)} (должно быть ~9-10)")


# ============================================================
# ПРИМЕР 5: Геометрия - расстояние и интерполяция
# ============================================================

print("\n" + "="*60)
print("ПРИМЕР 5: Геометрия на сфере")
print("="*60)

from core import normalize, d_geo, slerp, centroid
import numpy as np

# Две точки
a = normalize([1, 0, 0])
b = normalize([0, 1, 0])

# Расстояние
d = d_geo(a, b)
print(f"Расстояние между [1,0,0] и [0,1,0]: {np.degrees(d):.0f}°")

# Интерполяция
mid = slerp(a, b, 0.5)
print(f"Середина: {mid}")
print(f"Норма середины: {np.linalg.norm(mid):.4f} (должна быть 1)")

# Центроид
c = centroid(np.array([a, b]))
print(f"Центроид: {c}")


print("\n" + "="*60)
print("ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ!")
print("="*60)
