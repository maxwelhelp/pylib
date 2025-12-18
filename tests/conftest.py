"""
Pytest configuration - автоматически добавляет корень проекта в sys.path
"""
import sys
from pathlib import Path

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
