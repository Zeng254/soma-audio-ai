"""
SOMA Audio Separator Module
音频分离器模块 - 提供人声/伴奏分离、去混响、降噪等功能
"""

from .base import BaseSeparator, SeparationResult
from .demucs_separator import DemucsSeparator
from .msst_separator import MSSTSeparator

__all__ = [
    "BaseSeparator",
    "SeparationResult", 
    "DemucsSeparator",
    "MSSTSeparator",
]
