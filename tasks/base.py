"""
Base Model - общий интерфейс для всех моделей

Обеспечивает:
- Единый API (fit/predict)
- Безопасная сериализация (JSON + numpy)
- Версионирование моделей
"""

import json
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List
from pathlib import Path
from dataclasses import dataclass

# Версия формата сериализации
SERIALIZATION_VERSION = "1.0"


class BaseModel(ABC):
    """
    Абстрактный базовый класс для моделей.

    Все модели должны реализовать:
        - fit(): обучение
        - predict(): предсказание
        - _get_state(): получить состояние для сохранения
        - _set_state(): восстановить состояние

    Автоматически предоставляет:
        - save()/load(): сериализация в JSON
        - _is_fitted: флаг обученности
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> 'BaseModel':
        """Обучить модель. Должен установить _is_fitted = True."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Предсказание. Должен проверить _is_fitted."""
        pass

    @abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        """
        Получить состояние модели для сериализации.

        Returns:
            Словарь с параметрами (numpy arrays будут конвертированы автоматически)
        """
        pass

    @abstractmethod
    def _set_state(self, state: Dict[str, Any]) -> None:
        """
        Восстановить состояние модели.

        Args:
            state: словарь от _get_state() (numpy arrays уже восстановлены)
        """
        pass

    def _check_fitted(self) -> None:
        """Проверить что модель обучена."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} not fitted. Call fit() first.")

    def save(self, path: Union[str, Path]) -> None:
        """
        Сохранить модель в JSON файл.

        Формат безопасный — никакого pickle, только JSON + base64 для numpy.
        """
        self._check_fitted()

        state = self._get_state()

        # Конвертируем numpy в сериализуемый формат
        serializable = {
            '_version': SERIALIZATION_VERSION,
            '_class': self.__class__.__name__,
            '_state': _encode_state(state),
        }

        path = Path(path)
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """
        Загрузить модель из JSON файла.

        Безопасно — не выполняет произвольный код как pickle.
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        # Проверяем версию
        version = data.get('_version', '0.0')
        if version != SERIALIZATION_VERSION:
            # В будущем здесь можно добавить миграцию
            pass

        # Проверяем класс
        saved_class = data.get('_class')
        if saved_class != cls.__name__:
            raise ValueError(
                f"Model class mismatch: file contains {saved_class}, "
                f"but loading as {cls.__name__}"
            )

        # Декодируем состояние
        state = _decode_state(data['_state'])

        # Создаём и восстанавливаем модель
        model = cls.__new__(cls)
        model.__init__()
        model._set_state(state)
        model._is_fitted = True

        return model


def _encode_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Конвертировать состояние в JSON-сериализуемый формат."""
    encoded = {}
    for key, value in state.items():
        encoded[key] = _encode_value(value)
    return encoded


def _encode_value(value: Any) -> Any:
    """Конвертировать значение в JSON-сериализуемый формат."""
    if isinstance(value, np.ndarray):
        return {
            '_type': 'ndarray',
            '_dtype': str(value.dtype),
            '_shape': list(value.shape),
            '_data': value.tolist(),
        }
    elif isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    else:
        return value


def _decode_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Восстановить состояние из JSON формата."""
    decoded = {}
    for key, value in state.items():
        decoded[key] = _decode_value(value)
    return decoded


def _decode_value(value: Any) -> Any:
    """Восстановить значение из JSON формата."""
    if isinstance(value, dict):
        if value.get('_type') == 'ndarray':
            return np.array(value['_data'], dtype=value['_dtype']).reshape(value['_shape'])
        else:
            return {k: _decode_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_decode_value(v) for v in value]
    else:
        return value
