"""
Audio Separator - Unified interface for audio source separation

Provides a single entry point for various audio separation tasks:
- 2-stem separation (vocals/accompaniment)
- 4-stem separation (vocals/drums/bass/other)
- HPSS (Harmonic/Percussive Source Separation)
- Dereverberation

Usage:
    separator = AudioSeparator()
    
    # 2-stem separation
    vocals, accompaniment = separator.separate(audio, mode="2stems")
    
    # 4-stem separation
    vocals, drums, bass, other = separator.separate(audio, mode="4stems")
    
    # HPSS separation
    harmonic, percussive = separator.hpss(audio)
    
    # Dereverberation
    dry_audio = separator.dereverb(audio)
"""

import logging
from enum import Enum
from typing import Optional, Tuple, Union, List
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SeparationMode(str, Enum):
    """Separation mode enumeration."""
    TWO_STEMS = "2stems"       # vocals + accompaniment
    FOUR_STEMS = "4stems"      # vocals + drums + bass + other
    HPSS = "hpss"              # harmonic + percussive


class AudioSeparator:
    """
    Unified audio separator interface.
    
    Wraps multiple separation backends (Demucs, MSST) and provides
    additional signal processing methods (HPSS, dereverb).
    
    Attributes:
        backend: The separation backend to use ("demucs", "msst", "auto").
        device: Device to run separation on ("cpu", "cuda", "mps").
        sample_rate: Target sample rate for separation output.
    """
    
    def __init__(
        self,
        backend: str = "auto",
        device: Optional[str] = None,
        sample_rate: int = 44100,
    ):
        """
        Initialize the audio separator.
        
        Args:
            backend: Separation backend ("demucs", "msst", "auto").
                     "auto" will try demucs first, then fall back to HPSS.
            device: Device to run on ("cpu", "cuda", "mps").
            sample_rate: Target sample rate for output.
        """
        self.backend = backend
        self.device = device or self._get_default_device()
        self.sample_rate = sample_rate
        self._separator = None
        self._backend_available = {}
        
    def _get_default_device(self) -> str:
        """Get default compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def _load_separator(self, mode: SeparationMode):
        """
        Lazy-load the separation backend.
        
        Args:
            mode: The separation mode to prepare for.
        """
        if self._separator is not None:
            return
            
        if self.backend == "auto":
            # Try demucs first
            try:
                from src.separators.demucs_separator import DemucsSeparator
                self._separator = DemucsSeparator(
                    sample_rate=self.sample_rate,
                    device=self.device,
                )
                self._backend_available["demucs"] = True
                logger.info("Using Demucs backend for separation")
                return
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Demucs not available: {e}")
                
            # Fall back to MSST
            try:
                from src.separators.msst_separator import MSSTSeparator
                self._separator = MSSTSeparator(
                    sample_rate=self.sample_rate,
                    device=self.device,
                )
                self._backend_available["msst"] = True
                logger.info("Using MSST backend for separation")
                return
            except (ImportError, RuntimeError) as e:
                logger.warning(f"MSST not available: {e}")
                
            # No deep learning backend available
            logger.warning("No deep learning backend available. Using HPSS fallback.")
            self._backend_available["hpss_only"] = True
            
        elif self.backend == "demucs":
            from src.separators.demucs_separator import DemucsSeparator
            self._separator = DemucsSeparator(
                sample_rate=self.sample_rate,
                device=self.device,
            )
            self._backend_available["demucs"] = True
            
        elif self.backend == "msst":
            from src.separators.msst_separator import MSSTSeparator
            self._separator = MSSTSeparator(
                sample_rate=self.sample_rate,
                device=self.device,
            )
            self._backend_available["msst"] = True
    
    def separate(
        self,
        audio: Union[np.ndarray, str, Path],
        mode: Union[str, SeparationMode] = "2stems",
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, ...]:
        """
        Separate audio into stems.
        
        Args:
            audio: Input audio as numpy array or file path.
            mode: Separation mode ("2stems", "4stems", "hpss").
            sample_rate: Sample rate of input audio (required if audio is array).
            
        Returns:
            Tuple of separated audio arrays.
            - 2stems: (vocals, accompaniment)
            - 4stems: (vocals, drums, bass, other)
            - hpss: (harmonic, percussive)
        """
        mode = SeparationMode(mode)
        
        # Load audio from file if path provided
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio))
        elif sample_rate is None:
            sample_rate = self.sample_rate
            
        # Normalize audio shape
        audio = self._normalize_audio_shape(audio)
        
        if mode == SeparationMode.HPSS:
            return self.hpss(audio, sample_rate)
            
        # Load deep learning backend
        self._load_separator(mode)
        
        # Check if we have a deep learning backend
        if self._backend_available.get("hpss_only", False):
            logger.warning("Falling back to HPSS-based separation (lower quality)")
            return self._hpss_separation(audio, sample_rate, mode)
            
        # Use deep learning backend
        try:
            result = self._separator.separate_array(audio, sample_rate)
            
            if mode == SeparationMode.TWO_STEMS:
                # Combine drums/bass/other into accompaniment if needed
                vocals = result.vocals
                if result.accompaniment is not None:
                    accompaniment = result.accompaniment
                else:
                    # Combine other stems
                    accompaniment = np.zeros_like(vocals)
                    if result.drums is not None:
                        accompaniment += result.drums
                    if result.bass is not None:
                        accompaniment += result.bass
                    if result.other is not None:
                        accompaniment += result.other
                return vocals, accompaniment
                
            elif mode == SeparationMode.FOUR_STEMS:
                return (
                    result.vocals,
                    result.drums,
                    result.bass,
                    result.other,
                )
                
        except NotImplementedError:
            logger.warning("Backend not implemented, falling back to HPSS")
            return self._hpss_separation(audio, sample_rate, mode)
    
    def hpss(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
        kernel_size: Union[int, Tuple[int, int]] = 31,
        power: float = 2.0,
        mask_mode: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmonic-Percussive Source Separation (HPSS).
        
        Separates audio into harmonic (tonal) and percussive (transient) components.
        Uses median filtering on the spectrogram.
        
        Args:
            audio: Input audio as numpy array or file path.
            sample_rate: Sample rate of input audio.
            kernel_size: Size of median filter kernel. Can be int or (harmonic, percussive).
            power: Exponent for Wiener filtering.
            mask_mode: If True, use soft masking; if False, use hard masking.
            
        Returns:
            Tuple of (harmonic, percussive) audio arrays.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required for HPSS. Install with: uv add librosa")
        
        # Load audio from file if path provided
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio))
        elif sample_rate is None:
            sample_rate = self.sample_rate
            
        # Convert to mono for HPSS
        if audio.ndim == 2:
            if audio.shape[0] <= 2:
                # Channel first
                mono = np.mean(audio, axis=0)
            else:
                # Channel last
                mono = np.mean(audio, axis=1)
        else:
            mono = audio
            
        # Compute STFT
        stft = librosa.stft(mono, n_fft=4096, hop_length=1024)
        
        # Apply HPSS
        harmonic_mask, percussive_mask = librosa.decompose.hpss(
            stft,
            kernel_size=kernel_size,
            power=power,
            mask=mask_mode,
        )
        
        # Apply masks and inverse STFT
        harmonic_stft = stft * harmonic_mask
        percussive_stft = stft * percussive_mask
        
        harmonic = librosa.istft(harmonic_stft, hop_length=1024)
        percussive = librosa.istft(percussive_stft, hop_length=1024)
        
        # Match original length
        orig_len = len(mono)
        harmonic = harmonic[:orig_len]
        percussive = percussive[:orig_len]
        
        # Convert back to stereo if input was stereo
        if audio.ndim == 2 and audio.shape[0] == 2:
            harmonic = np.stack([harmonic, harmonic])
            percussive = np.stack([percussive, percussive])
        elif audio.ndim == 2 and audio.shape[1] == 2:
            harmonic = np.stack([harmonic, harmonic], axis=1)
            percussive = np.stack([percussive, percussive], axis=1)
            
        return harmonic, percussive
    
    def dereverb(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
        method: str = "spectral",
        reduction_db: float = 10.0,
        window_size: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Remove reverberation from audio.
        
        Uses spectral subtraction to estimate and remove the reverb tail.
        
        Args:
            audio: Input audio as numpy array or file path.
            sample_rate: Sample rate of input audio.
            method: Dereverberation method ("spectral", "wiener").
            reduction_db: Maximum reduction in decibels.
            window_size: FFT window size.
            hop_length: FFT hop length.
            
        Returns:
            Dereverberated audio array.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required for dereverb. Install with: uv add librosa")
        
        # Load audio from file if path provided
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio))
        elif sample_rate is None:
            sample_rate = self.sample_rate
            
        # Convert to mono for processing
        is_stereo = audio.ndim == 2
        if is_stereo:
            if audio.shape[0] <= 2:
                mono = np.mean(audio, axis=0)
            else:
                mono = np.mean(audio, axis=1)
        else:
            mono = audio.copy()
            
        # Compute STFT
        stft = librosa.stft(mono, n_fft=window_size, hop_length=hop_length)
        magnitude, phase = np.abs(stft), np.angle(stft)
        
        if method == "spectral":
            # Spectral subtraction based dereverberation
            # Estimate reverb as smoothed version of magnitude
            from scipy.ndimage import uniform_filter1d
            
            # Smooth magnitude to estimate reverb tail
            reverb_estimate = uniform_filter1d(
                magnitude, 
                size=10,  # Smooth over time
                axis=1
            )
            
            # Subtract reverb estimate with flooring
            reduction_linear = 10 ** (-reduction_db / 20)
            clean_magnitude = np.maximum(
                magnitude - reverb_estimate * 0.5,
                magnitude * reduction_linear
            )
            
        elif method == "wiener":
            # Wiener filter based dereverberation
            # Estimate reverb as delayed version of signal
            power = magnitude ** 2
            reverb_power = np.roll(power, shift=5, axis=1)  # Delay estimate
            reverb_power[:, :5] = 0  # Zero out initial frames
            
            # Wiener filter
            clean_power = power * np.maximum(1 - reverb_power / (power + 1e-10), 0.1)
            clean_magnitude = np.sqrt(clean_power)
            
        else:
            raise ValueError(f"Unknown dereverb method: {method}")
            
        # Reconstruct signal
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=hop_length, length=len(mono))
        
        # Convert back to stereo if needed
        if is_stereo:
            if audio.shape[0] == 2:
                clean_audio = np.stack([clean_audio, clean_audio])
            else:
                clean_audio = np.stack([clean_audio, clean_audio], axis=1)
                
        return clean_audio
    
    def _hpss_separation(
        self,
        audio: np.ndarray,
        sample_rate: int,
        mode: SeparationMode,
    ) -> Tuple[np.ndarray, ...]:
        """
        Fallback separation using HPSS.
        
        This is a lower-quality fallback when deep learning backends are unavailable.
        
        Args:
            audio: Input audio array.
            sample_rate: Sample rate.
            mode: Separation mode.
            
        Returns:
            Tuple of separated audio arrays.
        """
        harmonic, percussive = self.hpss(audio, sample_rate)
        
        if mode == SeparationMode.TWO_STEMS:
            # Approximate vocals as harmonic, accompaniment as percussive + some harmonic
            # This is a very rough approximation
            vocals = harmonic * 0.7 + percussive * 0.3
            accompaniment = harmonic * 0.3 + percussive * 0.7
            return vocals, accompaniment
            
        elif mode == SeparationMode.FOUR_STEMS:
            # Very rough 4-stem approximation
            # In practice, this is not useful - just for API compatibility
            vocals = harmonic * 0.5
            drums = percussive
            bass = harmonic * 0.3
            other = harmonic * 0.2
            return vocals, drums, bass, other
    
    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file path."""
        try:
            import librosa
            audio, sr = librosa.load(path, sr=None, mono=False)
            return audio, sr
        except ImportError:
            # Fall back to scipy
            from scipy.io import wavfile
            sr, audio = wavfile.read(path)
            audio = audio.astype(np.float32) / 32768.0
            return audio, sr
    
    def _normalize_audio_shape(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to channel-first format (channels, samples)."""
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # Detect format
            if audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
                # Already channel first
                return audio
            elif audio.shape[1] <= 2 and audio.shape[0] > audio.shape[1]:
                # Channel last, transpose
                return audio.T
        return audio
    
    def get_available_backends(self) -> List[str]:
        """
        Get list of available separation backends.
        
        Returns:
            List of available backend names.
        """
        available = []
        
        # Check demucs
        try:
            import demucs
            available.append("demucs")
        except ImportError:
            pass
            
        # Check librosa (for HPSS)
        try:
            import librosa
            available.append("hpss")
        except ImportError:
            pass
            
        return available
    
    def get_backend_info(self) -> dict:
        """
        Get information about the current backend configuration.
        
        Returns:
            Dictionary with backend information.
        """
        return {
            "backend": self.backend,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "available_backends": self.get_available_backends(),
            "separator_loaded": self._separator is not None,
        }
