"""
Lifting Client - HTTP клиент для Lifting API

Использование:
    >>> from client import LiftingClient
    >>> client = LiftingClient(api_key="your-key")
    >>> shapes = client.normalize(data)
"""

from .client import LiftingClient, LiftingClientError

__all__ = ['LiftingClient', 'LiftingClientError']
