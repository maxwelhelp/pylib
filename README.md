# Lifting Library v2.0

**Универсальная геометрическая библиотека для представления данных**

---

## Архитектура

```
lifting/
├── core/           # ЯДРО (не трогать)
│   ├── geometry    # Sphere, d_geo, centroid, slerp
│   ├── shape       # normalize, Shape
│   └── metrics     # d_eff, KL
│
├── transforms/     # КУБИКИ (комбинируй)
│   ├── spectral    # fft, dct, stft
│   ├── wavelet     # cwt, dwt
│   ├── analytic    # hilbert, teager, envelope
│   ├── sync        # SyncLifting ← КЛЮЧЕВОЙ
│   └── sequence    # kmer, ngram (DNA, text)
│
├── tasks/          # ЗАДАЧИ
│   ├── anomaly     # AnomalyDetector
│   └── classifier  # Classifier
│
└── domains/        # ПРИМЕРЫ (копируй)
    ├── ecg         # ECGAnalyzer
    ├── dna         # DNAAnalyzer
    ├── audio       # AudioAnalyzer
    └── template    # Шаблон для своего
```

---

## Уровни использования

### Новичок (3 строки)

```python
from lifting.domains import ECGAnalyzer

analyzer = ECGAnalyzer()
analyzer.fit_anomaly(ecg_signal, annotations)
anomalies = analyzer.detect_anomalies(test_signal)
```

### Практик (свой домен)

```python
from lifting.transforms import SyncLifting, fft, wavelet_energy
from lifting.tasks import AnomalyDetector
from lifting.core import normalize

class SeismicAnalyzer:
    def __init__(self):
        self.sync = SyncLifting(moments=[0, 1, 2])
        self.detector = AnomalyDetector(threshold_sigma=2.0)
    
    def extract_features(self, segment):
        m = self.sync.dominant_metrics(segment, fft)
        return {'fft_tc': m['tc'], 'fft_ts': m['ts']}
    
    def to_shape(self, features):
        return normalize(np.array(list(features.values())))
```

### Исследователь (новые кубики)

```python
from lifting.core import Sphere
from lifting.transforms import register_transform

# Новый transform
def my_transform(signal):
    return custom_processing(signal)

register_transform('my_transform', my_transform)

# Новая геометрия
class HyperbolicSpace(Sphere):
    def distance(self, x, y):
        # Poincaré distance
        ...
```

---

## Quick Start

### Базовые операции

```python
from lifting import normalize, d_geo, centroid, d_eff

# Данные → shape
signal = np.sin(np.linspace(0, 10, 256))
shape = normalize(signal)  # ||shape|| = 1

# Расстояние
d = d_geo(shape1, shape2)           # радианы
d_deg = np.degrees(d)               # градусы

# Центроид
c = centroid(shapes)                # среднее на сфере

# Эффективная размерность
dim = d_eff(shapes)                 # сколько "реальных" измерений
```

### Синхронный лифтинг

```python
from lifting import SyncLifting

sync = SyncLifting(moments=[0, 1, 2])

# FFT с временной локализацией
metrics = sync.compute(signal, np.fft.rfft)

# time_center[k] = где во времени энергия частоты k
# time_spread[k] = как долго длится частота k
print(metrics['time_center'])
print(metrics['time_spread'])
```

### Детекция аномалий

```python
from lifting import AnomalyDetector, normalize

# Shapes
train_shapes = [normalize(s) for s in train_signals]
test_shapes = [normalize(s) for s in test_signals]

# Детектор
detector = AnomalyDetector(threshold_sigma=2.0)
detector.fit(train_shapes)

# Детекция
results = detector.predict(test_shapes)
anomalies = [r for r in results if r.is_anomaly]
```

---

## Домены

### ECG

```python
from lifting.domains import ECGAnalyzer

analyzer = ECGAnalyzer(
    fs=360,                    # частота дискретизации
    window_ms=400,             # окно вокруг R-пика
    transforms=['fft', 'dct', 'hilbert', 'teager'],
    threshold_sigma=1.5,
)

analyzer.fit_anomaly(ecg, annotations)
results = analyzer.detect_anomalies(ecg)

# Sensitivity ~89%, Specificity ~95%
```

### DNA

```python
from lifting.domains import DNAAnalyzer

analyzer = DNAAnalyzer(
    k=4,                       # 4-mer → 256 features
    window_size=1000,          # bp per window
)

analyzer.fit_classifier(sequences, labels)
results = analyzer.classify(new_sequence)
```

### Audio / TTS

```python
from lifting.domains import TTSAnalyzer

analyzer = TTSAnalyzer(
    fs=22050,
    frame_ms=25,
    features=['sync_fft', 'wavelet'],
)

# Sync lifting видит ГДЕ энергия (position encoding)
shapes = analyzer.process(audio)
```

---

## Ключевые концепции

### Shape = точка на сфере

```
Данные → normalize → ||S|| = 1 → сфера S^{n-1}
```

Инвариантность к масштабу. Геодезическое расстояние = угол.

### Синхронный лифтинг

```
T(x)     → ЧТО (какие частоты)
T(x·t)   → ГДЕ (где во времени)
T(x·t²)  → КАК ДОЛГО (spread)
```

Обычный FFT не видит позицию. Sync lifting видит!

### Геометрия для всех

```
d_geo(S1, S2) = arccos(⟨S1, S2⟩)    # расстояние
centroid = normalize(mean)           # среднее
threshold = mean + k·std             # аномалии
```

Один метод для ECG, DNA, Audio, частиц, ...

---

## Результаты

| Домен | Задача | Метрика |
|-------|--------|---------|
| ECG | Arrhythmia detection | 89% sens, 95% spec |
| DNA | GC/AT classification | 100% accuracy |
| TTS | Position encoding | 0° → 43° с sync lifting |

---

## Установка

```python
# Копируем папку
import sys
sys.path.append('/path/to/lifting')

from lifting import ECGAnalyzer, d_geo, normalize
```

---

## License

MIT
