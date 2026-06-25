"""
SOMA - Smart Omnimedia Audio

AI driven audio processing workstation. Integrates voice separation, audio effects, audio generation and more.

Main modules:
- separators: Audio separator (Voice/accompaniment separation)
- effects: Audio effects processor (Equalizer, reverb, pitch shifting)
- converters: Format converter
- voice_converters: Voice converter (Dual engine architecture)
- pipeline: Processing pipeline
- config: Configuration center
- security: Security validation
- utils: Common utilities

Usage example:
    from src import Config, AudioLoader

    # Load configuration
    config = Config.load("~/.soma/config.json")

    # Load audio
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
"""

# Exception exports
from src.exceptions import (
    # Base class
    SOMAError,

    # Configuration exceptions
    ConfigError,
    ConfigLoadError,
    ConfigValidationError,
    ConfigTypeError,

    # Security exceptions
    SecurityError,
    PathTraversalError,
    AudioValidationError,
    ModelSecurityError,

    # Model exceptions
    ModelError,
    ModelLoadError,
    ModelNotFoundError,

    # Audio exceptions
    AudioError,
    AudioLoadError,
    AudioFormatError,

    # Separator exceptions
    SeparatorError,
    SeparationError,

    # Voice conversion exceptions
    VoiceConverterError,
    VoiceConversionError,

    # Utility functions
    is_soma_error,
    get_error_category,
    format_error,
)

__all__ = [
    # Version info
    "__version__",

    # Exceptions
    "SOMAError",
    "ConfigError",
    "ConfigLoadError",
    "ConfigValidationError",
    "ConfigTypeError",
    "SecurityError",
    "PathTraversalError",
    "AudioValidationError",
    "ModelSecurityError",
    "ModelError",
    "ModelLoadError",
    "ModelNotFoundError",
    "AudioError",
    "AudioLoadError",
    "AudioFormatError",
    "SeparatorError",
    "SeparationError",
    "VoiceConverterError",
    "VoiceConversionError",

    # Utility functions
    "is_soma_error",
    "get_error_category",
    "format_error",
]
