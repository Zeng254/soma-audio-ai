"""
SOMA Audio Separator Module
Audio separator module - Provides vocals/accompaniment separation, dereverberation, denoising and more
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
