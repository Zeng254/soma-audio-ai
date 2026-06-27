"""
SOMA Audio Separator Module
Audio separator module - Provides vocals/accompaniment separation, dereverberation, denoising and more
"""

from .base import BaseSeparator, SeparationResult
from .demucs_separator import DemucsSeparator
from .msst_separator import MSSTSeparator
from .audio_separator import AudioSeparator, SeparationMode

__all__ = [
    "BaseSeparator",
    "SeparationResult", 
    "DemucsSeparator",
    "MSSTSeparator",
    "AudioSeparator",
    "SeparationMode",
]
