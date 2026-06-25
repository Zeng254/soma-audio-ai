"""
SOMA Audio Effects Module
Audio effects module - Provides equalizer, reverb, pitch shifting and more
"""

from .base import BaseEffect, EffectResult
from .eq import Equalizer
from .reverb import Reverb
from .pitch import PitchShifter

__all__ = [
    "BaseEffect",
    "EffectResult",
    "Equalizer",
    "Reverb",
    "PitchShifter",
]
