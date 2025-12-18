"""
Internal imports helper.

Позволяет импортировать модули пакета независимо от контекста запуска.
"""

def _try_import(relative_path, absolute_path):
    """Try relative import first, fall back to absolute"""
    try:
        import importlib
        return importlib.import_module(relative_path)
    except ImportError:
        import importlib
        return importlib.import_module(absolute_path)


# Core imports
def get_normalize():
    try:
        from .core import normalize
    except ImportError:
        from core import normalize
    return normalize

def get_d_geo():
    try:
        from .core import d_geo
    except ImportError:
        from core import d_geo
    return d_geo

def get_centroid():
    try:
        from .core import centroid
    except ImportError:
        from core import centroid
    return centroid

def get_d_eff():
    try:
        from .core import d_eff
    except ImportError:
        from core import d_eff
    return d_eff
