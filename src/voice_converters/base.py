"""
Base Voice Converter - Voice conversion abstract base class
Defines common interface for all voice conversion engines
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import numpy as np

from src.exceptions import (
    SOMAError,
    SOMADependencyError,
    SOMAValidationError,
    SOMAConversionError,
)


class F0Method(Enum):
    """F0 extraction method"""
    PM = "pm"           # Cepstrum method，Fast but moderate accuracy
    DIO = "dio"         # DIO algorithm，Higher accuracy
    CREPE = "crepe"     # Deep learning method, most accurate but slowest
    CREPE_TINY = "crepe_tiny"  # Lightweight CREPE
    HARVEST = "harvest"  # Harvest algorithm, stable but slow
    RMVPE = "rmvpe"     # Resampling F0 prediction


class ConverterType(Enum):
    """Converter class type"""
    RVC = "rvc"
    SOVITS = "sovits"
    UNKNOWN = "unknown"


@dataclass
class ConversionParams:
    """
    Voice conversion common parameters
    
    These parameters are uniformly supported across all engines,
    Bottom layer will automatically map to specific parameters of each engine
    """
    # Pitch adjustment
    pitch_shift: float = 0.0          # Semitone shift (-24 to +24)
    pitch_algo: str = "rmvpe"         # Pitch algorithm (pm/dio/crepe/harvest/rmvpe)
    
    # Timbre control
    vpm: float = 0.5                  # Voicing period match (0.0-1.0)
    timbre_protection: float = 0.5    # Timbre protection (0.0-1.0)
    
    # Loudness control
    rms_mix: float = 0.5              # RMS loudness mix (0.0-1.0)
    loudness_match: bool = True       # Whether to match loudness
    
    # Quality control
    sample_rate: int = 40000          # OutputSample rate
    hop_length: int = 128              # Frame shift
    f0_factor: float = 1.0            # F0 scaling factor
    
    # DiffusionParameter (SoVITS)
    diffusion_steps: int = 10          # Diffusion steps
    diffusion_temperature: float = 1.0 # DiffusionTemperature
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pitch_shift": self.pitch_shift,
            "pitch_algo": self.pitch_algo,
            "vpm": self.vpm,
            "timbre_protection": self.timbre_protection,
            "rms_mix": self.rms_mix,
            "loudness_match": self.loudness_match,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "f0_factor": self.f0_factor,
            "diffusion_steps": self.diffusion_steps,
            "diffusion_temperature": self.diffusion_temperature,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversionParams":
        """FromDictionaryCreate"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversionResult:
    """
    Voice conversion result
    
    Unified return format, contains converted audio and metadata
    """
    audio: np.ndarray                 # Converted audio (samples, channels) or (samples,)
    sampling_rate: int                # Sample rate
    info: Dict[str, Any] = field(default_factory=dict)  # ConvertInfo
    
    # Quality metrics
    pitch_range: Optional[tuple] = None  # (min_hz, max_hz)
    duration: Optional[float] = None     # Duration (seconds)
    rms_db: Optional[float] = None       # RMS level (dB)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "audio_shape": self.audio.shape if self.audio is not None else None,
            "sampling_rate": self.sampling_rate,
            "info": self.info,
            "pitch_range": self.pitch_range,
            "duration": self.duration,
            "rms_db": self.rms_db,
        }


@dataclass
class ModelInfo:
    """ModelInfo"""
    name: str                          # ModelName
    type: ConverterType                 # Model class type
    version: Optional[str] = None      # Version number
    sample_rate: int = 40000          # Original sample rate
    description: Optional[str] = None  # Description
    file_path: Optional[str] = None    # File path
    config_path: Optional[str] = None  # Configuration filePath
    index_path: Optional[str] = None   # IndexFile path
    is_loaded: bool = False           # Whether loaded
    memory_usage: Optional[int] = None # Memory usage (bytes)
    
    def __repr__(self) -> str:
        return f"ModelInfo({self.type.value}: {self.name})"


class BaseVoiceConverter(ABC):
    """
    Voice converter base class
    
    Defines common interface for all voice conversion engines。
    Supports on-demand loading, VRAM management and graceful degradation.
    
    Common parameters:
    - pitch_shift: Semitone shift
    - vpm: Voicing period match
    - rms_mix: Loudness mix
    """
    
    # ClassAttribute：Supports f0 method
    SUPPORTED_F0_METHODS: List[F0Method] = []
    
    # Class attribute: whether index file is needed
    REQUIRE_INDEX: bool = False
    
    def __init__(self, device: Optional[str] = None):
        """
        InitializeVoice converter
        
        Args:
            device: Run device ('cpu', 'cuda', 'mps')
        """
        self.device = device or self._get_default_device()
        self._model = None
        self._config = None
        self._is_loaded = False
        self._model_info: Optional[ModelInfo] = None
    
    @abstractmethod
    def load_model(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        index_path: Optional[str] = None,
        **kwargs
    ) -> ModelInfo:
        """
        LoadModel
        
        Args:
            model_path: ModelFile path (.pth)
            config_path: Configuration filePath (.json)
            index_path: IndexFile path (.index)
            **kwargs: OtherParameter
            
        Returns:
            ModelInfo: ModelInfo
            
        Raises:
            FileNotFoundError: Model file does not exist
            ValueError: ModelFormatError
        """
        pass
    
    @abstractmethod
    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        params: Optional[ConversionParams] = None,
        **kwargs
    ) -> ConversionResult:
        """
        Execute voice conversion
        
        Args:
            audio: Input audio (samples,) or (channels, samples)
            sample_rate: Input sample rate
            params: ConvertParameter
            **kwargs: Other parameter override
            
        Returns:
            ConversionResult: Conversion result
        """
        pass
    
    @abstractmethod
    def unload(self):
        """
        Unload model, release VRAM
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model info"""
        pass
    
    def convert_file(
        self,
        input_path: str,
        output_path: str,
        params: Optional[ConversionParams] = None,
        **kwargs
    ) -> bool:
        """
        ConvertAudioFile
        
        Args:
            input_path: Input file path
            output_path: Output file path
            params: ConvertParameter
            **kwargs: OtherParameter
            
        Returns:
            bool: Whether successful
        """
        from src.utils.audio_io import AudioLoader, AudioSaver
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # LoadAudio
        loader = AudioLoader(target_sr=params.sample_rate if params else 40000)
        audio, sr = loader.load(input_path)
        
        # ExecuteConvert
        result = self.convert(audio, sr, params, **kwargs)
        
        # SaveAudio
        saver = AudioSaver(normalize=True)
        return saver.save(result.audio, output_path, result.sampling_rate)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded
    
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
    
    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Validate and normalize audio input
        
        Args:
            audio: Input audio
            
        Returns:
            Normalize audio array (samples,)
        """
        audio = np.array(audio, dtype=np.float32)
        
        # Process multiple channels -> mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        elif audio.ndim > 2:
            raise ValueError(f"Invalid audio shape: {audio.shape}")
        
        # Ensure it is 1D
        if audio.ndim != 1:
            audio = audio.flatten()
        
        return audio
    
    def _validate_params(self, params: Optional[ConversionParams]) -> ConversionParams:
        """Validate and normalize parameters"""
        if params is None:
            params = ConversionParams()
        
        # Limit pitch_shift range
        params.pitch_shift = max(-24, min(24, params.pitch_shift))
        
        # Limit vpm range
        params.vpm = max(0.0, min(1.0, params.vpm))
        
        # Limit rms_mix range
        params.rms_mix = max(0.0, min(1.0, params.rms_mix))
        
        return params
    
    def _create_result(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> ConversionResult:
        """Create standard result object"""
        # Calculate basic info
        duration = len(audio) / sample_rate if audio is not None else 0
        rms = self._calculate_rms(audio)
        
        return ConversionResult(
            audio=audio,
            sampling_rate=sample_rate,
            duration=duration,
            rms_db=rms,
            **kwargs
        )
    
    def _calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS level (dB)"""
        if audio is None or len(audio) == 0:
            return -float('inf')
        
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return 20 * np.log10(rms)
        return -float('inf')
    
    def _trim_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        top_db: int = 40
    ) -> np.ndarray:
        """
        Remove silent parts from audio
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            top_db: silence threshold (dB)
            
        Returns:
            Remove audio after silence
        """
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            return audio
        
        try:
            # Calculate energy
            hop_length = 512
            rms = librosa.feature.rms(
                y=audio,
                frame_length=2048,
                hop_length=hop_length
            )[0]
            
            # Find non-silent regions
            threshold = librosa.db_to_amplitude(-top_db)
            non_silent = np.where(rms > threshold)[0]
            
            if len(non_silent) == 0:
                return audio
            
            # Expand edges
            frame_start = max(0, non_silent[0] - 5)
            frame_end = min(len(rms), non_silent[-1] + 5)
            
            sample_start = frame_start * hop_length
            sample_end = min(len(audio), frame_end * hop_length + 1024)
            
            return audio[sample_start:sample_end]
            
        except Exception:
            return audio
    
    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "empty"
        return f"{self.__class__.__name__}({self.device}, {status})"
    
    def __enter__(self) -> "BaseVoiceConverter":
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit, auto cleanup"""
        self.unload()
        return False


class LazyImportMixin:
    """
    Delay import mixin class
    
    Provides lazy import for deep learning dependencies,
    Avoid checking all dependencies at module load time.
    Each subclass instance has independent cache to avoid interference.
    """
    
    # Subclass should set this attribute to list required modules
    REQUIRED_PACKAGES: List[str] = []
    
    # Cached checked package status - instance variable
    _package_cache: Dict[str, Optional[bool]] = {}
    
    def __init__(self):
        # Each instance has independent cache
        if not hasattr(self, '_instance_cache'):
            self._instance_cache: Dict[str, Optional[bool]] = {}
    
    @classmethod
    def _check_dependency(cls, package: str) -> bool:
        """
        Check if dependency is available (class level, check shared cache)
        
        Args:
            package: package name
            
        Returns:
            bool: whether available
        """
        if package in cls._package_cache:
            return cls._package_cache[package]
        
        try:
            __import__(package)
            cls._package_cache[package] = True
            return True
        except ImportError:
            cls._package_cache[package] = False
            return False
    
    @classmethod
    def _check_dependency_instance(cls, package: str) -> bool:
        """
        Check if dependency is available (instance level, each instance has independent cache)
        
        Args:
            package: package name
            
        Returns:
            bool: whether available
        """
        # First check instance cache
        if hasattr(cls, '_instance_cache') and package in cls._instance_cache:
            return cls._instance_cache[package]
        
        # CheckClassCache
        if package in cls._package_cache:
            result = cls._package_cache[package]
        else:
            try:
                __import__(package)
                cls._package_cache[package] = True
                result = True
            except ImportError:
                cls._package_cache[package] = False
                result = False
        
        # Store in instance cache
        if hasattr(cls, '_instance_cache'):
            cls._instance_cache[package] = result
        
        return result
    
    @classmethod
    def _lazy_import_module(cls, package: str, submodule: Optional[str] = None):
        """
        DelayImportModule
        
        Args:
            package: package name
            submodule: submodule name
            
        Returns:
            ImportModule
            
        Raises:
            ImportError: module not available
        """
        if not cls._check_dependency(package):
            raise ImportError(
                f"Required package '{package}' is not installed.\n"
                f"Please install it with: uv add {package}"
            )
        
        if submodule:
            return __import__(f"{package}.{submodule}", fromlist=[submodule])
        return __import__(package)
    
    @classmethod
    def _check_all_dependencies(cls) -> List[str]:
        """Check all dependencies, return missing package list"""
        missing = []
        for pkg in cls.REQUIRED_PACKAGES:
            if not cls._check_dependency(pkg):
                missing.append(pkg)
        return missing


class EngineCapability:
    """Engine capability description"""
    
    # Whether f0 adjustment is supported
    SUPPORTS_F0: bool = True
    
    # Whether timbre protection is supported
    SUPPORTS_TIMBRE_PROTECTION: bool = True
    
    # Whether diffusion mode is supported
    SUPPORTS_DIFFUSION: bool = False
    
    # Whether speaker embedding is supported
    SUPPORTS_SPEAKER_EMBEDDING: bool = False
    
    # MaximumSupportsSample rate
    MAX_SAMPLE_RATE: int = 48000
    
    # Recommended sample rate
    RECOMMENDED_SAMPLE_RATE: int = 40000
