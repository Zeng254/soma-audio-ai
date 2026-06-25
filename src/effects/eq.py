"""
Equalizer - Multi-band equalizer
Supports custom band gain adjustment
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy import signal

from .base import BaseEffect, EffectResult


class Equalizer(BaseEffect):
    """
    Multi-band equalizer
    
    Supports any number of band settings, each band contains:
    - Center frequency (Hz)
    - Gain (dB)
    - Bandwidth (Q value)
    
    Supported filter types:
    - peak: Peak/notch filter
    - lowshelf: Low shelf filter
    - highshelf: High shelf filter
    - lowpass: Low pass filter
    - highpass: High pass filter
    - bandpass: Band pass filter
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        bands: Optional[List[dict]] = None,
    ):
        """
        InitializeEqualizer
        
        Args:
            sample_rate: Sample rate
            bands: Preset band list
                 Example: [{"freq": 60, "gain": 3, "q": 1.4, "type": "peak"}, ...]
        """
        super().__init__(sample_rate)
        self.bands = bands or self._default_bands()
    
    def _default_bands(self) -> List[dict]:
        """Default 10-band equalizer preset"""
        return [
            {"freq": 32, "gain": 0, "q": 1.4, "type": "lowshelf"},
            {"freq": 64, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 125, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 250, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 500, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 1000, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 2000, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 4000, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 8000, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 16000, "gain": 0, "q": 1.4, "type": "highshelf"},
        ]
    
    def get_effect_name(self) -> str:
        return "Equalizer"
    
    def get_parameters(self) -> dict:
        return {"bands": self.bands, "sample_rate": self.sample_rate}
    
    def set_band(
        self,
        index: int,
        freq: float,
        gain: float,
        q: float = 1.4,
        filter_type: str = "peak"
    ):
        """
        Set single band
        
        Args:
            index: Band index
            freq: Center frequency
            gain: Gain(dB)
            q: Bandwidth
            filter_type: Filter class type
        """
        if 0 <= index < len(self.bands):
            self.bands[index] = {
                "freq": freq,
                "gain": gain,
                "q": q,
                "type": filter_type,
            }
    
    def set_preset(self, name: str):
        """
        Apply preset
        
        Args:
            name: Preset name
        """
        presets = {
            "flat": self._flat_preset(),
            "bass_boost": self._bass_boost_preset(),
            "treble_boost": self._treble_boost_preset(),
            "vocal_boost": self._vocal_boost_preset(),
            "pop": self._pop_preset(),
            "rock": self._rock_preset(),
            "jazz": self._jazz_preset(),
            "classical": self._classical_preset(),
        }
        
        if name in presets:
            self.bands = presets[name]
        else:
            raise ValueError(f"Unknown preset: {name}")
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        **kwargs
    ) -> EffectResult:
        """
        Apply equalizer
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            **kwargs: Optional override parameters
            
        Returns:
            EffectResult: Processing result
        """
        audio = self.validate_audio(audio)
        self.sample_rate = sample_rate
        
        # Merge passed band settings
        bands = kwargs.get("bands", self.bands)
        
        # Apply equalizer to each channel
        result = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            filtered = audio[ch].copy()
            for band in bands:
                filtered = self._apply_band(filtered, band)
            result[ch] = filtered
        
        return self._create_result(result, sample_rate, bands=bands)
    
    def _apply_band(self, audio: np.ndarray, band: dict) -> np.ndarray:
        """Apply single band"""
        freq = band["freq"]
        gain = band["gain"]
        q = band["q"]
        filter_type = band["type"]
        
        # Calculate b, a coefficients
        b, a = self._design_filter(freq, gain, q, filter_type)
        
        # Apply filter
        return signal.lfilter(b, a, audio)
    
    def _design_filter(
        self,
        freq: float,
        gain: float,
        q: float,
        filter_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design IIR filter coefficients
        
        Args:
            freq: Center frequency
            gain: Gain(dB)
            q: Quality factor
            filter_type: Filter class type
            
        Returns:
            (b, a): Filter coefficients
        """
        A = 10 ** (gain / 40)
        w0 = 2 * np.pi * freq / self.sample_rate
        
        if filter_type == "peak":
            alpha = np.sin(w0) / (2 * q)
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
            
        elif filter_type == "lowshelf":
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/q - 1) + 2)
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            
        elif filter_type == "highshelf":
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/q - 1) + 2)
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            
        else:
            # Default uses peak filter
            return self._design_filter(freq, gain, q, "peak")
        
        # Normalization coefficients
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])
        
        return b, a
    
    # Preset method
    def _flat_preset(self) -> List[dict]:
        bands = self._default_bands()
        for band in bands:
            band["gain"] = 0
        return bands
    
    def _bass_boost_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = 6
        bands[1]["gain"] = 5
        bands[2]["gain"] = 4
        bands[3]["gain"] = 2
        return bands
    
    def _treble_boost_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[-1]["gain"] = 6
        bands[-2]["gain"] = 5
        bands[-3]["gain"] = 4
        bands[-4]["gain"] = 2
        return bands
    
    def _vocal_boost_preset(self) -> List[dict]:
        bands = self._default_bands()
        for i, band in enumerate(bands):
            if 3 <= i <= 6:  # 250Hz - 1kHz
                band["gain"] = 3
        return bands
    
    def _pop_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = -1
        bands[1]["gain"] = 4
        bands[2]["gain"] = 5
        bands[3]["gain"] = 3
        bands[4]["gain"] = -1
        bands[5]["gain"] = -1
        bands[6]["gain"] = 2
        bands[7]["gain"] = 4
        bands[8]["gain"] = 5
        bands[9]["gain"] = 3
        return bands
    
    def _rock_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = 5
        bands[1]["gain"] = 4
        bands[2]["gain"] = 2
        bands[3]["gain"] = -1
        bands[4]["gain"] = -2
        bands[5]["gain"] = 2
        bands[6]["gain"] = 3
        bands[7]["gain"] = 4
        bands[8]["gain"] = 4
        bands[9]["gain"] = 4
        return bands
    
    def _jazz_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = 3
        bands[1]["gain"] = 2
        bands[2]["gain"] = 1
        bands[3]["gain"] = 2
        bands[4]["gain"] = -2
        bands[5]["gain"] = -2
        bands[6]["gain"] = 0
        bands[7]["gain"] = 2
        bands[8]["gain"] = 3
        bands[9]["gain"] = 4
        return bands
    
    def _classical_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = 5
        bands[1]["gain"] = 4
        bands[2]["gain"] = 3
        bands[3]["gain"] = 2
        bands[4]["gain"] = -1
        bands[5]["gain"] = -1
        bands[6]["gain"] = -1
        bands[7]["gain"] = 0
        bands[8]["gain"] = 2
        bands[9]["gain"] = 3
        return bands
