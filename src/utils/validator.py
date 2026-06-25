"""
Validator - Parameter validation tools
Provides audio processing parameter validation
"""

from typing import Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

import logging
from src.exceptions import SOMAError

logger = logging.getLogger(__name__)


class ValidationError(SOMAError):
    """Validation error"""
    pass


class AudioValidator:
    """
    Audio parameter validator
    
    Validate audio processing input parameters
    """
    
    # SupportsSample rate
    VALID_SAMPLE_RATES = {
        8000, 11025, 16000, 22050, 32000, 
        44100, 48000, 88200, 96000, 176400, 192000
    }
    
    # Supports channel count
    VALID_CHANNELS = {1, 2, 6, 8}  # mono, stereo, 5.1, 7.1
    
    # Supported audio formats
    VALID_FORMATS = {
        "wav", "mp3", "flac", "ogg", "aac", 
        "m4a", "wma", "aiff", "amr", "opus"
    }
    
    @classmethod
    def validate_audio_path(cls, path: str, must_exist: bool = True) -> Path:
        """
        ValidateAudioFile path
        
        Args:
            path: File path
            must_exist: Whether must exist
            
        Returns:
            Path Object
            
        Raises:
            ValidationError: Path invalid
        """
        if not path:
            raise ValidationError("File path cannot be empty")
        
        file_path = Path(path)
        
        if must_exist and not file_path.exists():
            raise ValidationError(f"File does not exist: {path}")
        
        if must_exist and not file_path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        
        # Check extension
        suffix = file_path.suffix[1:].lower()
        if suffix not in cls.VALID_FORMATS:
            raise ValidationError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(sorted(cls.VALID_FORMATS))}"
            )
        
        return file_path
    
    @classmethod
    def validate_sample_rate(cls, sample_rate: int) -> int:
        """
        ValidateSample rate
        
        Args:
            sample_rate: Sample rate
            
        Returns:
            Validated sample rate
            
        Raises:
            ValidationError: Sample rate invalid
        """
        if not isinstance(sample_rate, int):
            raise ValidationError(f"Sample rate must be integer, got {type(sample_rate)}")
        
        if sample_rate <= 0:
            raise ValidationError(f"Sample rate must be positive, got {sample_rate}")
        
        if sample_rate not in cls.VALID_SAMPLE_RATES:
            # Warning but do not raise error
            logger.warning(f"Non-standard sample rate {sample_rate}Hz")
        
        return sample_rate
    
    @classmethod
    def validate_channels(cls, channels: int) -> int:
        """
        Validate channel count
        
        Args:
            channels: Channel count
            
        Returns:
            Validated channel count
            
        Raises:
            ValidationError: Channel count invalid
        """
        if not isinstance(channels, int):
            raise ValidationError(f"Channels must be integer, got {type(channels)}")
        
        if channels <= 0:
            raise ValidationError(f"Channels must be positive, got {channels}")
        
        if channels not in cls.VALID_CHANNELS:
            raise ValidationError(
                f"Unsupported channels: {channels}. "
                f"Supported: {', '.join(map(str, sorted(cls.VALID_CHANNELS)))}"
            )
        
        return channels
    
    @classmethod
    def validate_audio_array(cls, audio: np.ndarray) -> np.ndarray:
        """
        ValidateAudioarray
        
        Args:
            audio: Audio data
            
        Returns:
            Validated audio array
            
        Raises:
            ValidationError: Audio data invalid
        """
        if audio is None:
            raise ValidationError("Audio data cannot be None")
        
        if not isinstance(audio, np.ndarray):
            try:
                audio = np.array(audio)
            except Exception:
                raise ValidationError("Audio data cannot be converted to numpy array")
        
        if audio.ndim == 0:
            raise ValidationError("Audio data is scalar, expected array")
        
        if audio.ndim > 2:
            raise ValidationError(f"Audio data has too many dimensions: {audio.ndim}")
        
        if len(audio) == 0:
            raise ValidationError("Audio data is empty")
        
        return audio
    
    @classmethod
    def validate_frequency(cls, freq: float, min_freq: float = 0, max_freq: float = 20000) -> float:
        """
        ValidateFrequencyParameter
        
        Args:
            freq: Frequency value
            min_freq: MinimumFrequency
            max_freq: MaximumFrequency
            
        Returns:
            Validated frequency
            
        Raises:
            ValidationError: Frequency invalid
        """
        if not isinstance(freq, (int, float)):
            raise ValidationError(f"Frequency must be number, got {type(freq)}")
        
        if freq < min_freq or freq > max_freq:
            raise ValidationError(
                f"Frequency {freq}Hz out of range [{min_freq}, {max_freq}]Hz"
            )
        
        return float(freq)
    
    @classmethod
    def validate_gain(cls, gain: float, min_gain: float = -60, max_gain: float = 60) -> float:
        """
        ValidateGainParameter
        
        Args:
            gain: Gain value (dB)
            min_gain: MinimumGain
            max_gain: MaximumGain
            
        Returns:
            Validated gain
            
        Raises:
            ValidationError: Gain invalid
        """
        if not isinstance(gain, (int, float)):
            raise ValidationError(f"Gain must be number, got {type(gain)}")
        
        if gain < min_gain or gain > max_gain:
            raise ValidationError(
                f"Gain {gain}dB out of range [{min_gain}, {max_gain}]dB"
            )
        
        return float(gain)


class EffectParameterValidator:
    """
    Effect processor parameter validator
    """
    
    @classmethod
    def validate_eq_params(cls, params: dict) -> dict:
        """
        ValidateEqualizerParameter
        
        Args:
            params: ParameterDictionary
            
        Returns:
            Validated parameters
            
        Raises:
            ValidationError: Parameter invalid
        """
        if "bands" in params:
            bands = params["bands"]
            if not isinstance(bands, list):
                raise ValidationError("EQ bands must be a list")
            
            for i, band in enumerate(bands):
                if not isinstance(band, dict):
                    raise ValidationError(f"Band {i} must be a dictionary")
                
                if "freq" in band:
                    AudioValidator.validate_frequency(band["freq"], 20, 20000)
                
                if "gain" in band:
                    AudioValidator.validate_gain(band["gain"], -20, 20)
                
                if "q" in band:
                    q = band["q"]
                    if not 0.1 <= q <= 10:
                        raise ValidationError(f"Band Q must be in [0.1, 10], got {q}")
        
        return params
    
    @classmethod
    def validate_reverb_params(cls, params: dict) -> dict:
        """
        ValidateReverb parameters
        
        Args:
            params: ParameterDictionary
            
        Returns:
            Validated parameters
        """
        # Validate room_size
        if "room_size" in params:
            room_size = params["room_size"]
            if not 0.0 <= room_size <= 1.0:
                raise ValidationError(f"room_size must be in [0, 1], got {room_size}")
        
        # Validate damping
        if "damping" in params:
            damping = params["damping"]
            if not 0.0 <= damping <= 1.0:
                raise ValidationError(f"damping must be in [0, 1], got {damping}")
        
        # Validate wet_level and dry_level
        for key in ["wet_level", "dry_level"]:
            if key in params:
                level = params[key]
                if not 0.0 <= level <= 1.0:
                    raise ValidationError(f"{key} must be in [0, 1], got {level}")
        
        return params
    
    @classmethod
    def validate_pitch_params(cls, params: dict) -> dict:
        """
        ValidatePitchParameter
        
        Args:
            params: ParameterDictionary
            
        Returns:
            Validated parameters
        """
        # Validate semitones
        if "semitones" in params:
            semitones = params["semitones"]
            if not -24 <= semitones <= 24:
                raise ValidationError(f"semitones must be in [-24, 24], got {semitones}")
        
        # Validate cents
        if "cents" in params:
            cents = params["cents"]
            if not -100 <= cents <= 100:
                raise ValidationError(f"cents must be in [-100, 100], got {cents}")
        
        return params


class PipelineValidator:
    """
    Pipeline parameter validator
    """
    
    @classmethod
    def validate_node_config(cls, config: dict) -> dict:
        """
        ValidateNodeConfiguration
        
        Args:
            config: NodeConfiguration
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: Configuration invalid
        """
        required_fields = ["name", "type", "params"]
        
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field: {field}")
        
        # ValidateNode types
        valid_types = {"separator", "effect", "converter", "filter", "custom"}
        if config["type"] not in valid_types:
            raise ValidationError(
                f"Invalid node type: {config['type']}. "
                f"Valid types: {', '.join(valid_types)}"
            )
        
        return config


# ============== Convenience utility functions ==============

def validate_pitch_shift(value: float, min_val: float = -24.0, max_val: float = 24.0) -> float:
    """
    Validate pitch shift value
    
    Args:
        value: Semitone shift value
        min_val: Minimum value (default -24)
        max_val: Maximum value (default +24)
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: Value out of range
    """
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"Pitch shift must be a number, got {type(value)}")
    
    if not min_val <= value <= max_val:
        raise ValidationError(f"Pitch shift {value} out of range [{min_val}, {max_val}]")
    
    return value


def validate_duration(value: float, min_val: float = 0.1, max_val: float = 3600.0) -> float:
    """
    Validate audio duration
    
    Args:
        value: Duration (seconds)
        min_val: Minimum value (default 0.1)
        max_val: Maximum value (default 3600)
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: Value out of range
    """
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"Duration must be a number, got {type(value)}")
    
    if not min_val <= value <= max_val:
        raise ValidationError(f"Duration {value}s out of range [{min_val}, {max_val}]s")
    
    return value


def validate_model_path(path: str) -> str:
    """
    ValidateModelPath
    
    Args:
        path: ModelFile path
        
    Returns:
        Validated path
        
    Raises:
        ValidationError: Path invalid
    """
    if not path:
        raise ValidationError("Model path cannot be empty")
    
    if not isinstance(path, str):
        raise ValidationError(f"Model path must be string, got {type(path)}")
    
    # Check extension
    valid_extensions = ['.pth', '.pt', '.onnx', '.ckpt', '.safetensors']
    ext = os.path.splitext(path)[1].lower()
    if ext not in valid_extensions:
        raise ValidationError(f"Invalid model extension: {ext}. Valid: {valid_extensions}")
    
    return path


def validate_audio_format(value: str) -> str:
    """
    ValidateAudioFormat
    
    Args:
        value: AudioFormatString
        
    Returns:
        Validated format
        
    Raises:
        ValidationError: Format invalid
    """
    if not value:
        raise ValidationError("Audio format cannot be empty")
    
    valid_formats = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma'}
    value = value.lower().lstrip('.')
    
    if value not in valid_formats:
        raise ValidationError(f"Invalid audio format: {value}. Valid: {valid_formats}")
    
    return value


def validate_float(value: float, min_val: float = None, max_val: float = None) -> float:
    """
    ValidateFloat
    
    Args:
        value: Value
        min_val: Minimum value (optional)
        max_val: Maximum value (optional)
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: Value invalid or out of range
    """
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"Value must be convertible to float, got {type(value)}")
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"Value {value} below minimum {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"Value {value} above maximum {max_val}")
    
    return value
    
    @classmethod
    def validate_pipeline_config(cls, config: dict) -> dict:
        """
        Validate pipeline configuration
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Validated configuration
        """
        if "nodes" not in config:
            raise ValidationError("Pipeline config must contain 'nodes' field")
        
        if not isinstance(config["nodes"], list):
            raise ValidationError("Pipeline nodes must be a list")
        
        for i, node_config in enumerate(config["nodes"]):
            try:
                cls.validate_node_config(node_config)
            except ValidationError as e:
                raise ValidationError(f"Node {i}: {str(e)}")
        
        return config


def validate(value: Any, rules: dict) -> Tuple[bool, Optional[str]]:
    """
    Generic validation function
    
    Args:
        value: Value to validate
        rules: Validation rules
        
    Returns:
        (is_valid, error_message)
    """
    try:
        if "type" in rules:
            expected_type = rules["type"]
            if not isinstance(value, expected_type):
                return False, f"Expected {expected_type.__name__}, got {type(value).__name__}"
        
        if "min" in rules and value < rules["min"]:
            return False, f"Value {value} is less than minimum {rules['min']}"
        
        if "max" in rules and value > rules["max"]:
            return False, f"Value {value} is greater than maximum {rules['max']}"
        
        if "choices" in rules and value not in rules["choices"]:
            return False, f"Value {value} not in allowed choices: {rules['choices']}"
        
        if "regex" in rules:
            import re
            if not re.match(rules["regex"], str(value)):
                return False, f"Value does not match pattern: {rules['regex']}"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


# =============================================================================
# Convenience standalone function - for testing and external calls
# =============================================================================

def validate_sample_rate(sample_rate: Union[int, str]) -> int:
    """
    ValidateSample rate
    
    Args:
        sample_rate: Sample rate value
        
    Returns:
        Validated sample rate (integer)
        
    Raises:
        ValidationError: Sample rate invalid
    """
    # Process string input
    if isinstance(sample_rate, str):
        try:
            sample_rate = int(sample_rate)
        except ValueError:
            raise ValidationError(f"Invalid sample rate: {sample_rate}")
    
    if not isinstance(sample_rate, int):
        raise ValidationError(f"Sample rate must be an integer, got {type(sample_rate).__name__}")
    
    if sample_rate < 8000:
        raise ValidationError(f"Sample rate too low: {sample_rate} (min: 8000)")
    
    if sample_rate > 192000:
        raise ValidationError(f"Sample rate too high: {sample_rate} (max: 192000)")
    
    return sample_rate


def validate_pitch_shift(semitones: int) -> int:
    """
    Validate pitch offset value
    
    Args:
        semitones: semitone offset
        
    Returns:
        Validate passed offset value
        
    Raises:
        ValidationError: offset value out of range
    """
    if not isinstance(semitones, int):
        raise ValidationError(f"Pitch shift must be an integer, got {type(semitones).__name__}")
    
    if semitones < -24 or semitones > 24:
        raise ValidationError(f"Pitch shift out of range: {semitones} (valid: -24 to 24)")
    
    return semitones


def validate_duration(duration: float) -> float:
    """
    Validate audio duration
    
    Args:
        duration: duration (seconds)
        
    Returns:
        Validate passed duration
        
    Raises:
        ValidationError: duration invalid
    """
    if not isinstance(duration, (int, float)):
        raise ValidationError(f"Duration must be a number, got {type(duration).__name__}")
    
    duration = float(duration)
    
    if duration <= 0:
        raise ValidationError(f"Duration must be positive: {duration}")
    
    return duration


def validate_model_path(path: str) -> str:
    """
    ValidateModelFile path
    
    Args:
        path: ModelFile path
        
    Returns:
        Validate passed path
        
    Raises:
        ValidationError: path invalid or extension not supported
    """
    SUPPORTED_EXTENSIONS = {'.pth', '.pt', '.onnx', '.pkl', '.joblib'}
    
    if not path or not isinstance(path, str):
        raise ValidationError("Model path must be a non-empty string")
    
    path_obj = Path(path)
    ext = path_obj.suffix.lower()
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported model extension: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    
    return path


def validate_audio_format(fmt: str) -> str:
    """
    ValidateAudioFormat
    
    Args:
        fmt: AudioFormatString
        
    Returns:
        Format uppercase form
        
    Raises:
        ValidationError: format not supported
    """
    if not fmt or not isinstance(fmt, str):
        raise ValidationError("Audio format must be a non-empty string")
    
    fmt_lower = fmt.lower().strip().lstrip('.')
    valid_formats = {"wav", "mp3", "flac", "ogg", "m4a"}
    
    if fmt_lower not in valid_formats:
        raise ValidationError(
            f"Unsupported audio format: {fmt}. "
            f"Supported: {', '.join(sorted(valid_formats))}"
        )
    
    return fmt_lower.upper()


def validate_float(
    value: Union[int, float, str],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> float:
    """
    Validate float value
    
    Args:
        value: Value to validate
        min_val: minimum value
        max_val: maximum value
        
    Returns:
        Validate passed float value
        
    Raises:
        ValidationError: Value invalid or out of range
    """
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValidationError(f"Invalid float value: {value}")
    
    if not isinstance(value, (int, float)):
        raise ValidationError(f"Value must be numeric, got {type(value).__name__}")
    
    value = float(value)
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"Value {value} is less than minimum {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"Value {value} is greater than maximum {max_val}")
    
    return value
