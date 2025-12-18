"""
Domains - Готовые анализаторы для разных областей

Это ПРИМЕРЫ и УДОБСТВО, не ядро.
Пользователь может создать свой домен из transforms + tasks.

Доступные домены:
- ECGAnalyzer: ЭКГ сигналы, детекция аритмий
- DNAAnalyzer: ДНК последовательности, классификация
- AudioAnalyzer: Аудио, речь
- TTSAnalyzer: Text-to-Speech фичи

Шаблон для своего домена:
- TemplateDomainAnalyzer: скопируй и измени
"""

from .base import BaseDomainAnalyzer
from .ecg import ECGAnalyzer
from .dna import DNAAnalyzer
from .audio import AudioAnalyzer, TTSAnalyzer
from .template import TemplateDomainAnalyzer, SeismicAnalyzer

__all__ = [
    'BaseDomainAnalyzer',
    'ECGAnalyzer',
    'DNAAnalyzer',
    'AudioAnalyzer',
    'TTSAnalyzer',
    'TemplateDomainAnalyzer',
    'SeismicAnalyzer',
]
