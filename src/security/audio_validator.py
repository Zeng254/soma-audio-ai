"""
SOMA Security module - AudioFile validation

Provides audio file security checks, including:
- File format validation
- Sample rate range check
- Duration limit check
- FileSizeCheck
- MIME type detection
- Audio metadata validation

Usage:
    from src.security.audio_validator import AudioValidator, validate_audio

    # Create validator
    validator = AudioValidator()

    # ValidateAudioFile
    result = validator.validate("/path/to/audio.wav")

    # Use convenience function
    result = validate_audio("/path/to/audio.wav", max_duration=300)
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from src.config import AudioUtilsDefaults
from src.security.path_validator import safe_path
from src.exceptions import AudioFormatError as BaseAudioFormatError
from src.exceptions import AudioProcessingError as BaseAudioProcessingError

logger = logging.getLogger(__name__)


class AudioFormatError(BaseAudioFormatError):
    """AudioFormatErrorException"""
    pass


class AudioValidationError(BaseAudioProcessingError):
    """AudioValidateFailException"""
    pass


class AudioFormat(Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    AAC = "aac"
    WMA = "wma"
    AIFF = "aiff"
    UNKNOWN = "unknown"


@dataclass
class AudioMetadata:
    """Audio metadata"""
    format: AudioFormat
    sample_rate: int
    channels: int
    bit_depth: Optional[int]
    duration: float  # seconds
    bitrate: Optional[int]  # bps
    file_size: int  # Bytes
    codec: Optional[str] = None
    is_lossless: bool = False


@dataclass
class AudioValidationResult:
    """Audio validation result"""
    is_valid: bool
    metadata: Optional[AudioMetadata] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AudioValidator:
    """
    Audio file validator

    Features:
    - Validate if file is valid audio file
    - Check if file format is supported
    - Validate sample rate range
    - Check duration limit
    - ValidateFileSize
    - Read audio metadata

    Usage example:
        validator = AudioValidator(
            allowed_formats=[AudioFormat.WAV, AudioFormat.MP3],
            max_duration=3600,
            max_file_size_mb=500
        )

        result = validator.validate("/path/to/audio.wav")
        if result.is_valid:
            print(f"Sample rate: {result.metadata.sample_rate}")
        else:
            print(f"ValidateFail: {result.errors}")
    """

    # Common audio file magic numbers
    MAGIC_BYTES: Dict[AudioFormat, bytes] = {
        AudioFormat.WAV: b'RIFF',
        AudioFormat.FLAC: b'fLaC',
        AudioFormat.OGG: b'OggS',
        AudioFormat.MP3: b'\xff\xfb',  # MP3 frame sync
        AudioFormat.M4A: b'ftyp',       # MP4/M4A
    }

    # MIME type mapping
    MIME_TYPES: Dict[str, AudioFormat] = {
        'audio/wav': AudioFormat.WAV,
        'audio/x-wav': AudioFormat.WAV,
        'audio/mpeg': AudioFormat.MP3,
        'audio/mp3': AudioFormat.MP3,
        'audio/flac': AudioFormat.FLAC,
        'audio/ogg': AudioFormat.OGG,
        'audio/x-m4a': AudioFormat.M4A,
        'audio/aac': AudioFormat.AAC,
    }

    def __init__(
        self,
        allowed_formats: Optional[List[AudioFormat]] = None,
        min_sample_rate: int = 8000,
        max_sample_rate: int = 192000,
        min_duration: float = 0.1,  # Minimum 100ms
        max_duration: float = 3600,  # Maximum 1 Hour
        max_file_size_mb: int = 500,
        defaults: Optional[AudioUtilsDefaults] = None
    ):
        """
        Initialize audio validator

        Args:
            allowed_formats: Allowed audio format list
            min_sample_rate: MinimumSample rate
            max_sample_rate: MaximumSample rate
            min_duration: Minimum audio duration (seconds)
            max_duration: Maximum audio duration (seconds)
            max_file_size_mb: MaximumFileSize（MB）
            defaults: Audio tools default configuration
        """
        self.defaults = defaults or AudioUtilsDefaults()

        # Use configuration or parameters
        self.allowed_formats = allowed_formats or [
            AudioFormat(f) for f in self.defaults.allowed_audio_formats
        ] if hasattr(self.defaults, 'allowed_audio_formats') else [
            AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FLAC,
            AudioFormat.OGG, AudioFormat.M4A
        ]

        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration or self.defaults.max_duration_seconds
        self.max_file_size_bytes = (max_file_size_mb or self.defaults.max_file_size_mb) * 1024 * 1024
        self.min_file_size_bytes = self.defaults.min_file_size_bytes

    def validate(
        self,
        path: Union[str, Path],
        check_metadata: bool = True
    ) -> AudioValidationResult:
        """
        ValidateAudioFile

        Args:
            path: AudioFile path
            check_metadata: Whether to read metadata for validation

        Returns:
            AudioValidationResult Validation result
        """
        result = AudioValidationResult(is_valid=False)
        
        # 0. First convert to Path object, check if file exists
        # If file does not exist, return is_valid=False instead of raising exception
        if isinstance(path, str):
            path = Path(path)
        
        # 1. Check if file exists
        if not path.exists():
            result.errors.append(f"File does not exist: {path}")
            return result
        
        # 2. Check path security (only for existing files)
        try:
            path = safe_path(path)
        except Exception as e:
            result.errors.append(f"PathSecurityCheckFail: {str(e)}")
            return result

        # 2. CheckFileSize
        self._check_file_size(path, result)
        if result.errors:
            return result

        # 3. DetectionFileFormat
        detected_format = self._detect_format(path)
        result.metadata = AudioMetadata(
            format=detected_format,
            sample_rate=0,
            channels=0,
            bit_depth=None,
            duration=0.0,
            bitrate=None,
            file_size=path.stat().st_size
        )

        # 4. Check if format is allowed
        if not self._is_format_allowed(detected_format):
            result.errors.append(
                f"Unsupported audio formats: {detected_format.value}, "
                f"Allowed formats: {[f.value for f in self.allowed_formats]}"
            )
            return result

        # 5. Read metadata (if needed)
        if check_metadata:
            self._check_metadata(path, result)

        if not result.errors:
            result.is_valid = True

        return result

    def _check_file_size(
        self,
        path: Path,
        result: AudioValidationResult
    ) -> None:
        """CheckFileSize"""
        file_size = path.stat().st_size

        if file_size < self.min_file_size_bytes:
            result.errors.append(
                f"File too small ({file_size} bytes), minimum requirement: {self.min_file_size_bytes} bytes"
            )

        if file_size > self.max_file_size_bytes:
            result.errors.append(
                f"File too large ({file_size / 1024 / 1024:.1f} MB), "
                f"Maximum allowed: {self.max_file_size_bytes / 1024 / 1024:.1f} MB"
            )

    def _detect_format(self, path: Path) -> AudioFormat:
        """DetectionAudioFileFormat"""
        # 1. Guess by extension
        ext = path.suffix.lower().lstrip('.')

        # Special processing
        if ext in ['mp2', 'mp3']:
            ext = 'mp3'
        elif ext in ['aif', 'aifc']:
            ext = 'aiff'

        try:
            return AudioFormat(ext)
        except ValueError:
            pass

        # 2. Detect by magic number
        try:
            with open(path, 'rb') as f:
                header = f.read(16)

                for fmt, magic in self.MAGIC_BYTES.items():
                    if header.startswith(magic):
                        return fmt

                # MP3 special detection (may have ID3 header)
                if len(header) >= 3 and header[0:2] in [b'\xff\xfb', b'\xff\xf3', b'\xff\xf2']:
                    return AudioFormat.MP3

        except Exception as e:
            logger.warning(f"Cannot read file header: {e}")

        return AudioFormat.UNKNOWN

    def _is_format_allowed(self, fmt: AudioFormat) -> bool:
        """Check if format is allowed"""
        if fmt == AudioFormat.UNKNOWN:
            return False
        return fmt in self.allowed_formats

    def _check_metadata(
        self,
        path: Path,
        result: AudioValidationResult
    ) -> None:
        """Check audio metadata"""
        try:
            import soundfile as sf

            # ReadAudioInfo
            info = sf.info(str(path))

            # Update metadata
            result.metadata.sample_rate = info.samplerate
            result.metadata.channels = info.channels
            result.metadata.duration = info.duration
            result.metadata.codec = info.format

            # Detect if lossless formats
            result.metadata.is_lossless = info.format.lower() in ['wav', 'flac', 'aiff']

            # Get sub-format info
            if hasattr(info, 'subtype'):
                if info.subtype in ['PCM_16', 'PCM_24', 'PCM_32']:
                    result.metadata.bit_depth = int(info.subtype.split('_')[1])

            # CheckSample rate
            if result.metadata.sample_rate < self.min_sample_rate:
                result.errors.append(
                    f"Sample rate too low ({result.metadata.sample_rate} Hz), "
                    f"Minimum requirement: {self.min_sample_rate} Hz"
                )
            elif result.metadata.sample_rate > self.max_sample_rate:
                result.warnings.append(
                    f"Sample rate too high ({result.metadata.sample_rate} Hz), "
                    f"May affect processing speed"
                )

            # Check duration
            if result.metadata.duration < self.min_duration:
                result.errors.append(
                    f"Audio duration too short ({result.metadata.duration:.2f}s), "
                    f"Minimum requirement: {self.min_duration}s"
                )
            elif result.metadata.duration > self.max_duration:
                result.errors.append(
                    f"Audio duration too long ({result.metadata.duration:.1f}s), "
                    f"Maximum allowed: {self.max_duration}s"
                )

            # Check channel count
            if result.metadata.channels > 2:
                result.warnings.append(
                    f"Audio is {result.metadata.channels} channels, will be converted to stereo"
                )

        except ImportError:
            result.warnings.append("soundfile not installed, cannot validate audio metadata")
        except Exception as e:
            result.errors.append(f"Cannot read audio metadata: {e}")

    def get_metadata(
        self,
        path: Union[str, Path]
    ) -> Optional[AudioMetadata]:
        """
        Get audio metadata

        Args:
            path: AudioFile path

        Returns:
            AudioMetadata or None
        """
        result = self.validate(path, check_metadata=True)
        if not result.is_valid:
            return None
        # If soundfile was not available and metadata is still default zeros,
        # try stdlib wave module as fallback for WAV files
        if result.metadata and result.metadata.sample_rate == 0:
            self._read_metadata_fallback(path, result)
        return result.metadata

    def _read_metadata_fallback(
        self,
        path: Path,
        result: AudioValidationResult
    ) -> None:
        """Fallback metadata reading using stdlib wave/aifc modules"""
        try:
            import wave
            with wave.open(str(path), 'rb') as wf:
                result.metadata.sample_rate = wf.getframerate()
                result.metadata.channels = wf.getnchannels()
                result.metadata.bit_depth = wf.getsampwidth() * 8
                n_frames = wf.getnframes()
                result.metadata.duration = n_frames / wf.getframerate()
                result.metadata.is_lossless = True
                result.metadata.codec = 'pcm'
        except Exception:
            pass  # Cannot read metadata


# Global validator instance
_default_validator: Optional[AudioValidator] = None


def get_audio_validator() -> AudioValidator:
    """Get global audio validator instance"""
    global _default_validator
    if _default_validator is None:
        _default_validator = AudioValidator()
    return _default_validator


def validate_audio(
    path: Union[str, Path],
    max_duration: Optional[float] = None,
    check_metadata: bool = True
) -> AudioValidationResult:
    """
    Convenient audio validation function

    Args:
        path: AudioFile path
        max_duration: maximum duration limit (seconds)
        check_metadata: whether to check metadata

    Returns:
        AudioValidationResult

    Example:
        result = validate_audio("/path/to/audio.wav")
        if result.is_valid:
            print(f"Audio valid, sample rate: {result.metadata.sample_rate}")
    """
    # Create new validator instance each time to avoid thread safety issues
    validator = AudioValidator(max_duration=max_duration)
    return validator.validate(path, check_metadata)
