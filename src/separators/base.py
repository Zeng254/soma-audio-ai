"""
Base Separator - Audio separator base class
Defines common interface and abstract methods for all separators
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import numpy as np

from src.utils.audio.validation import validate_audio_input as _validate_audio_input


@dataclass
class SeparationResult:
    """Audio separation result"""
    vocals: Optional[np.ndarray] = None      # Voice
    accompaniment: Optional[np.ndarray] = None  # Accompaniment
    drums: Optional[np.ndarray] = None      # Drums
    bass: Optional[np.ndarray] = None       # bass
    other: Optional[np.ndarray] = None       # Other
    sample_rate: int = 44100                 # Sample rate
    
    def get_track(self, name: str) -> Optional[np.ndarray]:
        """Get separated track by name"""
        track_map = {
            "vocals": self.vocals,
            "accompaniment": self.accompaniment,
            "drums": self.drums,
            "bass": self.bass,
            "other": self.other,
        }
        return track_map.get(name)
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            "vocals": self.vocals is not None,
            "accompaniment": self.accompaniment is not None,
            "drums": self.drums is not None,
            "bass": self.bass is not None,
            "other": self.other is not None,
            "sample_rate": self.sample_rate,
        }


class BaseSeparator(ABC):
    """
    Audio separator base class
    
    Defines common interface for audio separation tasks, supports:
    - Voice/accompaniment separation
    - Multi-track separation (drums, bass, other)
    - Dereverberation
    - Denoising
    """
    
    def __init__(self, sample_rate: int = 44100, device: Optional[str] = None):
        """
        InitializeSeparator
        
        Args:
            sample_rate: ObjectSample rate
            device: Run device ('cpu', 'cuda', 'mps')
        """
        self.sample_rate = sample_rate
        self.device = device or self._get_default_device()
    
    @abstractmethod
    def separate(self, audio_path: str, **kwargs) -> SeparationResult:
        """
        ExecuteAudioSeparation
        
        Args:
            audio_path: Input audio file path
            **kwargs: OtherParameter
            
        Returns:
            SeparationResult: Separation result
        """
        pass
    
    @abstractmethod
    def separate_array(self, audio: np.ndarray, sample_rate: int = 44100, **kwargs) -> SeparationResult:
        """
        Perform separation on audio array
        
        Args:
            audio: Audio data (samples, channels) or (channels, samples)
            sample_rate: Sample rate
            **kwargs: OtherParameter
            
        Returns:
            SeparationResult: Separation result
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """GetModelName"""
        pass
    
    @abstractmethod
    def get_available_tracks(self) -> List[str]:
        """Get separable track list"""
        pass
    
    def _get_default_device(self) -> str:
        """Get default run device"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def validate_audio_input(self, audio: np.ndarray) -> np.ndarray:
        """
        Validate and normalize audio input.
        
        Uses the common validation function from src.utils.audio.validation.
        
        Args:
            audio: Input audio
            
        Returns:
            Normalized audio array (channels, samples)
        """
        return _validate_audio_input(audio)
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio level
        
        Args:
            audio: Input audio
            target_db: Target decibel value
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_linear = 10 ** (target_db / 20)
            audio = audio * (target_linear / max_val)
        return audio
