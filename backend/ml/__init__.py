"""
ML Package for Intent Classification and Semantic Inference.
Keep package root lightweight to avoid circular import issues during startup.
"""

from .config import config, MLConfig

__all__ = [
    "config",
    "MLConfig",
]

