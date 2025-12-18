"""
Sequence Transforms - для ДНК, текста, категориальных данных

k-mer, n-gram и другие преобразования последовательностей.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional
from itertools import product


def kmer_frequencies(sequence: str, k: int = 4) -> NDArray:
    """
    k-mer частоты для ДНК последовательности
    
    Args:
        sequence: строка из {A, C, G, T}
        k: размер k-мера
        
    Returns:
        Вектор частот размера 4^k
        
    Examples:
        >>> freq = kmer_frequencies("ATGCATGC", k=2)
        >>> len(freq)  # 16 (4² возможных 2-меров)
    """
    # Все возможные k-меры
    alphabet = 'ACGT'
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(kmers)}
    
    # Подсчёт
    counts = np.zeros(len(kmers), dtype=np.float64)
    
    sequence = sequence.upper()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in kmer_to_idx:
            counts[kmer_to_idx[kmer]] += 1
    
    # Нормализация
    total = counts.sum()
    if total > 0:
        counts = counts / total
    
    return counts


def kmer_spectrum(sequence: str, k_range: range = range(2, 6)) -> NDArray:
    """
    Спектр k-меров для нескольких k
    
    Args:
        sequence: ДНК строка
        k_range: диапазон k (по умолчанию 2-5)
        
    Returns:
        Конкатенация частот для всех k
    """
    spectra = []
    for k in k_range:
        spectra.append(kmer_frequencies(sequence, k))
    return np.concatenate(spectra)


def gc_content(sequence: str) -> float:
    """
    GC-content (доля G+C)
    
    Returns:
        Число от 0 до 1
    """
    sequence = sequence.upper()
    gc = sum(1 for c in sequence if c in 'GC')
    return gc / len(sequence) if len(sequence) > 0 else 0.0


def dna_to_numeric(sequence: str) -> NDArray:
    """
    ДНК → числовая последовательность
    
    A=0, C=1, G=2, T=3
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([mapping.get(c.upper(), 0) for c in sequence])


def one_hot_encode(sequence: str, alphabet: str = 'ACGT') -> NDArray:
    """
    One-hot encoding последовательности
    
    Args:
        sequence: входная строка
        alphabet: алфавит
        
    Returns:
        [len(sequence), len(alphabet)] матрица
    """
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    result = np.zeros((len(sequence), len(alphabet)), dtype=np.float64)
    
    for i, c in enumerate(sequence):
        c_upper = c.upper()
        if c_upper in char_to_idx:
            result[i, char_to_idx[c_upper]] = 1.0
    
    return result


def ngram_frequencies(text: str, n: int = 2, 
                      vocab: Optional[List[str]] = None) -> NDArray:
    """
    n-gram частоты для текста
    
    Args:
        text: входной текст
        n: размер n-грамма
        vocab: словарь (если None, строится автоматически)
        
    Returns:
        Вектор частот
    """
    # Токенизация (простая — по символам)
    tokens = list(text.lower())
    
    # Сбор n-граммов
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    # Подсчёт
    if vocab is None:
        vocab = sorted(set(ngrams))
    
    ngram_to_idx = {ng: i for i, ng in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float64)
    
    for ng in ngrams:
        if ng in ngram_to_idx:
            counts[ngram_to_idx[ng]] += 1
    
    # Нормализация
    total = counts.sum()
    if total > 0:
        counts = counts / total
    
    return counts


def transition_matrix(sequence: str, alphabet: str = 'ACGT') -> NDArray:
    """
    Матрица переходов (Markov chain)
    
    T[i,j] = P(следующий символ = j | текущий = i)
    
    Returns:
        [len(alphabet), len(alphabet)] матрица
    """
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    n = len(alphabet)
    
    counts = np.zeros((n, n), dtype=np.float64)
    
    sequence = sequence.upper()
    for i in range(len(sequence) - 1):
        c1 = sequence[i]
        c2 = sequence[i + 1]
        if c1 in char_to_idx and c2 in char_to_idx:
            counts[char_to_idx[c1], char_to_idx[c2]] += 1
    
    # Нормализация по строкам
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # избегаем деления на 0
    return counts / row_sums


def sequence_complexity(sequence: str, k: int = 3) -> float:
    """
    Linguistic complexity (уникальные k-меры / возможные k-меры)
    
    Низкая для повторяющихся последовательностей.
    """
    unique_kmers = set()
    for i in range(len(sequence) - k + 1):
        unique_kmers.add(sequence[i:i+k])
    
    # Максимум возможных k-меров
    max_possible = min(len(sequence) - k + 1, 4 ** k)  # для ДНК
    
    return len(unique_kmers) / max_possible if max_possible > 0 else 0
