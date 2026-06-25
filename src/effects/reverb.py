"""
Reverb - Reverb effect processor
Implements room reverb, hall reverb, plate reverb and other types
"""

from typing import Optional, Tuple
import numpy as np
from scipy import signal

from .base import BaseEffect, EffectResult


class Reverb(BaseEffect):
    """
    Reverb effect processor
    
    Based on Schroeder/Moorer model digital reverb implementation.
    
    Reverb parameters:
    - room_size: Room size (0-1)
    - damping: Damping (0-1)
    - wet_level: Reverb signal level (0-1)
    - dry_level: Dry signal level (0-1)
    - width: StereoWidth (0-1)
    
    Reverb types:
    - room: Room reverb
    - hall: hall reverb
    - plate: Plate reverb
    - cathedral: Cathedral reverb
    """
    
    # Comb filter delay line length (samples @ 44100Hz)
    COMB_DELAYS = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
    
    # All-pass filter delay line length
    ALLPASS_DELAYS = [225, 556, 441, 341]
    
    def __init__(
        self,
        sample_rate: int = 44100,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3,
        dry_level: float = 0.7,
        width: float = 1.0,
        reverb_type: str = "room",
    ):
        """
        Initialize reverb processor
        
        Args:
            sample_rate: Sample rate
            room_size: Room size (0-1)
            damping: High frequency damping (0-1)
            wet_level: Reverb signal level
            dry_level: Dry signal level
            width: StereoWidth
            reverb_type: Reverb class type
        """
        super().__init__(sample_rate)
        self.room_size = max(0.0, min(1.0, room_size))
        self.damping = max(0.0, min(1.0, damping))
        self.wet_level = max(0.0, min(1.0, wet_level))
        self.dry_level = max(0.0, min(1.0, dry_level))
        self.width = max(0.0, min(1.0, width))
        self.reverb_type = reverb_type
        
        # InitializeFilterStatus
        self._comb_buffers = None
        self._allpass_buffers = None
        self._init_buffers()
    
    def _init_buffers(self):
        """Initialize delay line buffer"""
        # Comb filter buffer
        self._comb_buffers = [
            np.zeros(delay) for delay in self.COMB_DELAYS
        ]
        self._comb_indices = [0] * len(self.COMB_DELAYS)
        
        # All-pass filter buffer
        self._allpass_buffers = [
            np.zeros(delay) for delay in self.ALLPASS_DELAYS
        ]
        self._allpass_indices = [0] * len(self.ALLPASS_DELAYS)
    
    def get_effect_name(self) -> str:
        return f"Reverb-{self.reverb_type}"
    
    def get_parameters(self) -> dict:
        return {
            "room_size": self.room_size,
            "damping": self.damping,
            "wet_level": self.wet_level,
            "dry_level": self.dry_level,
            "width": self.width,
            "reverb_type": self.reverb_type,
        }
    
    def _adjust_for_type(self):
        """Adjust parameters based on reverb class type"""
        presets = {
            "room": {
                "room_size": 0.4,
                "damping": 0.6,
                "comb_feedback": 0.7,
            },
            "hall": {
                "room_size": 0.8,
                "damping": 0.3,
                "comb_feedback": 0.85,
            },
            "plate": {
                "room_size": 0.6,
                "damping": 0.8,
                "comb_feedback": 0.75,
            },
            "cathedral": {
                "room_size": 0.95,
                "damping": 0.2,
                "comb_feedback": 0.92,
            },
        }
        
        if self.reverb_type in presets:
            preset = presets[self.reverb_type]
            self.room_size = preset["room_size"]
            self.damping = preset["damping"]
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        **kwargs
    ) -> EffectResult:
        """
        Apply reverb effect
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            **kwargs: Optional parameter override
            
        Returns:
            EffectResult: Processing result
        """
        audio = self.validate_audio(audio)
        self.sample_rate = sample_rate
        
        # UpdateParameter
        for key in ["room_size", "damping", "wet_level", "dry_level", "width"]:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        
        # Handle reverb_type change
        if "reverb_type" in kwargs:
            self.reverb_type = kwargs["reverb_type"]
            self._adjust_for_type()
        
        # Reinitialize buffer (e.g. if sample rate changes)
        if self._comb_buffers is None or len(self._comb_buffers[0]) != int(self.COMB_DELAYS[0] * sample_rate / 44100):
            self._init_buffers()
        
        # Scale delay to adapt to sample rate
        scale = sample_rate / 44100
        
        # Process each channel
        is_stereo = audio.shape[0] >= 2
        
        if is_stereo:
            # StereoProcess
            left = self._process_channel(audio[0], scale)
            right = self._process_channel(audio[1], scale)
            
            # Apply stereo width
            if self.width < 1.0:
                mid = (left + right) / 2
                left = mid + self.width * (left - mid)
                right = mid + self.width * (right - mid)
            
            result = np.vstack([left, right])
        else:
            # Process mono
            result = self._process_channel(audio[0], scale)
            result = result[np.newaxis, :]
        
        return self._create_result(result, sample_rate)
    
    def _process_channel(self, audio: np.ndarray, scale: float) -> np.ndarray:
        """Process single channel"""
        output = np.zeros_like(audio)
        
        # Calculate comb filter feedback
        feedback = 0.7 + (self.room_size * 0.28)
        
        # Process each sample point
        for i, sample in enumerate(audio):
            reverb_sample = 0.0
            
            # Parallel comb filter
            for j, delay in enumerate(self.COMB_DELAYS):
                actual_delay = int(delay * scale)
                if actual_delay > 0 and actual_delay <= len(self._comb_buffers[j]):
                    idx = (self._comb_indices[j] - actual_delay) % len(self._comb_buffers[j])
                    reverb_sample += self._comb_buffers[j][idx]
            
            # Normalization
            reverb_sample /= len(self.COMB_DELAYS)
            
            # Write to buffer
            for j in range(len(self._comb_buffers)):
                self._comb_buffers[j][self._comb_indices[j]] = (
                    sample + reverb_sample * feedback * (1 - self.damping * 0.5)
                )
                self._comb_indices[j] = (self._comb_indices[j] + 1) % len(self._comb_buffers[j])
            
            # Cascade all-pass filter
            temp = reverb_sample
            for j, delay in enumerate(self.ALLPASS_DELAYS):
                actual_delay = int(delay * scale)
                if actual_delay > 0 and actual_delay <= len(self._allpass_buffers[j]):
                    idx = (self._allpass_indices[j] - actual_delay) % len(self._allpass_buffers[j])
                    temp2 = self._allpass_buffers[j][idx]
                    self._allpass_buffers[j][self._allpass_indices[j]] = temp
                    self._allpass_indices[j] = (self._allpass_indices[j] + 1) % len(self._allpass_buffers[j])
                    temp = -0.5 * temp + temp2 + 0.5 * self._allpass_buffers[j][idx]
            
            output[i] = sample * self.dry_level + temp * self.wet_level
        
        return output
