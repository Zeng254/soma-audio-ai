"""
Base Effect - Audio effects processor base class
Defines common interface for all audio effects
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


@dataclass
class EffectResult:
    """Audio effects result"""
    audio: np.ndarray                    # Processed audio
    sample_rate: int = 44100             # Sample rate
    parameters_used: Optional[dict] = None  # UsesParameter
    metadata: Optional[dict] = None     # Metadata
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            "audio_shape": self.audio.shape if self.audio is not None else None,
            "sample_rate": self.sample_rate,
            "parameters_used": self.parameters_used,
            "metadata": self.metadata,
        }


class BaseEffect(ABC):
    """
    Audio effects processor base class
    
    All audio effects processors should inherit this class and implement process method.
    Supports direct audio array processing and file processing.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize audio effects processor
        
        Args:
            sample_rate: Sample rate
        """
        self.sample_rate = sample_rate
    
    @abstractmethod
    def process(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 44100,
        **kwargs
    ) -> EffectResult:
        """
        ProcessAudio
        
        Args:
            audio: Input audio data
            sample_rate: Sample rate
            **kwargs: Effect specific parameters
            
        Returns:
            EffectResult: Processing result
        """
        pass
    
    @abstractmethod
    def get_effect_name(self) -> str:
        """GetEffectName"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> dict:
        """GetEffectParameter"""
        pass
    
    def validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Validate audio input
        
        Args:
            audio: Input audio
            
        Returns:
            Validated audio array (channels, samples)
        """
        audio = np.array(audio, dtype=np.float32)
        
        # Process mono
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        elif audio.ndim == 2:
            # Detect format: if first dim is small (<=8), likely channel_first
            # If second dim is small (<=8), likely channel_last
            dim0, dim1 = audio.shape
            if dim0 <= 8 and dim1 > dim0 * 2:
                # Already channel_first (channels, samples)
                pass
            elif dim1 <= 8 and dim0 > dim1 * 2:
                # Channel_last (samples, channels), transpose
                audio = audio.T
            # Otherwise assume channel_first
        
        return audio
    
    def apply_with_bypass(
        self,
        audio: np.ndarray,
        sample_rate: int,
        bypass: bool = False,
        **kwargs
    ) -> EffectResult:
        """
        Apply effect, supports bypass mode
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            bypass: Whether bypassed
            **kwargs: OtherParameter
            
        Returns:
            EffectResult: Processing result
        """
        if bypass:
            return EffectResult(
                audio=self.validate_audio(audio),
                sample_rate=sample_rate,
                parameters_used={"bypass": True},
            )
        
        return self.process(audio, sample_rate, **kwargs)
    
    def _create_result(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> EffectResult:
        """Create standard result object"""
        return EffectResult(
            audio=audio,
            sample_rate=sample_rate,
            parameters_used=self.get_parameters(),
            metadata={**kwargs} if kwargs else None,
        )
