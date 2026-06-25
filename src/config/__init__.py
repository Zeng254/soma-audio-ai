"""
SOMA Configuration center

Provides unified configuration management. Supports hierarchical override and type-safe access.

Main components:
- Config: Main configuration class
- get_config: Get global configuration instance
- init_config: Initialize configuration

Hierarchical override order:
1. Default values (defaults.py)
2. Configuration file (JSON/YAML)
3. User input (override at runtime)

Usage example:
    from src.config import get_config, Config

    # Get global configuration
    config = get_config()

    # Get configuration value
    sample_rate = config.get("audio_utils.default_sample_rate")
    device = config.get("separators.device", default="cuda")

    # Set configuration value
    config.set("separators.device", "cuda")

    # Save configuration
    config.save()

    # Direct import of default values
    from src.config.defaults import SeparatorDefaults, VoiceConverterDefaults
"""

from .defaults import (
    DEFAULT_CONFIG,
    SomaDefaults,
    SeparatorDefaults,
    VoiceConverterDefaults,
    EffectsDefaults,
    ConverterDefaults,
    AudioUtilsDefaults,
    SecurityDefaults,
    LoggingDefaults,
)

from .config import (
    Config,
    get_config,
    get_config_path,
    init_config,
)

from src.exceptions import ConfigError, ConfigLoadError, ConfigValidationError, ConfigTypeError

__all__ = [
    # Configuration class
    "Config",

    # Exceptions
    "ConfigError",
    "ConfigLoadError",
    "ConfigValidationError",
    "ConfigTypeError",

    # Default configuration
    "DEFAULT_CONFIG",
    "SomaDefaults",
    "SeparatorDefaults",
    "VoiceConverterDefaults",
    "EffectsDefaults",
    "ConverterDefaults",
    "AudioUtilsDefaults",
    "SecurityDefaults",
    "LoggingDefaults",

    # Factory functions
    "get_config",
    "get_config_path",
    "init_config",
]
