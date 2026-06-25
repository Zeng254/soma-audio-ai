"""
SOMA Configuration Center - Main configuration class

Provides unified configuration management, supports:
- Loading and saving from JSON/YAML files
- Hierarchical override: Default values < Configuration file < User input
- Type-safe get/set methods
- Configuration validation and auto-completion
"""

import os
import json
import copy
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar, Type
from dataclasses import is_dataclass, asdict

from .defaults import DEFAULT_CONFIG
from src.exceptions import ConfigError, ConfigLoadError, ConfigValidationError

T = TypeVar('T')

logger = logging.getLogger(__name__)


def _safe_asdict(obj):
    """Safely convert dataclass to dictionary (handle nesting)"""
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _safe_asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_asdict(item) for item in obj]
    if is_dataclass(obj):
        return {k: _safe_asdict(v) for k, v in asdict(obj).items()}
    return obj


class Config:
    """
    SOMA Configuration Management Class

    Supports hierarchical override mechanism:
    1. Default values (defaults.py)
    2. Configuration file (JSON/YAML)
    3. User input (runtime override)

    Example:
        # Load configuration
        config = Config.load("~/.soma/config.json")

        # Get value
        sample_rate = config.get("audio_utils.default_sample_rate")
        device = config.get("separators.device", default="cuda")

        # Set value
        config.set("separators.device", "cuda")

        # Save configuration
        config.save()

        # Reset to default
        config.reset()
    """

    def __init__(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        user_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize configuration

        Args:
            base_config: Base configuration (default configuration)
            user_config: User configuration (overrides base configuration)
        """
        if base_config is None:
            self._base = copy.deepcopy(DEFAULT_CONFIG)
        elif isinstance(base_config, dict):
            self._base = base_config
        elif is_dataclass(base_config) and not isinstance(base_config, type):
            # If it's a dataclass instance, convert to dictionary
            self._base = asdict(base_config)
        else:
            self._base = copy.deepcopy(DEFAULT_CONFIG)
        self._user = user_config or {}
        self._path: Optional[Path] = None
        self._loaded = False
        
        # Apply user configuration to base configuration
        if self._user:
            self._apply_user_config(self._base, self._user)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        auto_create: bool = True
    ) -> "Config":
        """
        Load configuration from file

        Args:
            path: Configuration file path
            auto_create: Whether to auto-create default configuration if file doesn't exist

        Returns:
            Config instance
        """
        path = Path(path).expanduser()
        config = cls()

        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if path.suffix in ['.yaml', '.yml']:
                        import yaml
                        config._user = yaml.safe_load(f) or {}
                    else:
                        config._user = json.load(f)
                config._path = path
                config._loaded = True
                logger.info(f"Configuration loaded from {path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration file: {e}, using default configuration")
                config._user = {}
        elif auto_create:
            # Auto-create default configuration file
            config._path = path
            config.save()
            logger.info(f"Created default configuration file: {path}")
        else:
            logger.info("Using default configuration")

        return config

    @classmethod
    def _apply_user_config(
        cls,
        base: Any,
        user: Dict[str, Any]
    ) -> None:
        """Apply user configuration to base configuration object (supports dictionary and dataclass)"""
        # Process dataclass object
        if is_dataclass(base) and not isinstance(base, type):
            base_dict = asdict(base) if hasattr(base, '__dataclass_fields__') else {}
            for key, value in user.items():
                if hasattr(base, key):
                    current = getattr(base, key)
                    if isinstance(current, dict) and isinstance(value, dict):
                        cls._apply_dict(current, value)
                    elif value is not None:
                        # Type-safe conversion
                        try:
                            if isinstance(current, bool):
                                if isinstance(value, str):
                                    value = value.lower() in ('true', '1', 'yes')
                                else:
                                    value = bool(value)
                            elif isinstance(current, int) and isinstance(value, (int, float)):
                                value = int(value)
                            elif isinstance(current, float) and isinstance(value, (int, float)):
                                value = float(value)
                            setattr(base, key, value)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Configuration type conversion failed {key}: {e}")
        # ProcessDictionaryObject
        elif isinstance(base, dict):
            for key, value in user.items():
                if key in base:
                    if isinstance(base[key], dict) and isinstance(value, dict):
                        cls._apply_dict(base[key], value)
                    elif value is not None:
                        base[key] = value

    @classmethod
    def _apply_dict(
        cls,
        base_dict: Dict[str, Any],
        user_dict: Dict[str, Any]
    ) -> None:
        """Apply dictionary values to dictionary object"""
        for key, value in user_dict.items():
            if key in base_dict:
                if isinstance(base_dict[key], dict) and isinstance(value, dict):
                    cls._apply_dict(base_dict[key], value)
                elif value is not None:
                    # Type-safe conversion
                    try:
                        current = base_dict[key]
                        if isinstance(current, bool):
                            if isinstance(value, str):
                                value = value.lower() in ('true', '1', 'yes')
                            else:
                                value = bool(value)
                        elif isinstance(current, int) and isinstance(value, (int, float)):
                            value = int(value)
                        elif isinstance(current, float) and isinstance(value, (int, float)):
                            value = float(value)
                        elif isinstance(current, list) and not isinstance(value, list):
                            value = [value]
                        base_dict[key] = value
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Configuration type conversion failed {key}: {e}")

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file

        Args:
            path: Save path, defaults to the path used during load
        """
        save_path = Path(path or self._path).expanduser() if self._path or path else None

        if not save_path:
            raise ConfigError("Save path not specified")

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert configuration to dictionary
        config_dict = self.to_dict()

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.suffix in ['.yaml', '.yml']:
                    import yaml
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            raise ConfigError(f"Failed to save configuration: {e}")

    def get(
        self,
        key: str,
        default: Optional[T] = None,
        value_type: Optional[Type[T]] = None
    ) -> Optional[T]:
        """
        Get configuration value

        Args:
            key: Configuration key, supports dot-separated path, e.g. "separators.device"
            default: Default value
            value_type: Expected type

        Returns:
            Configuration value

        Example:
            config.get("audio_utils.default_sample_rate")
            config.get("separators.device", default="cuda")
        """
        # First try to get from user configuration
        value = self._get_nested(self._user, key)

        # If not in user configuration, get from base configuration
        if value is None:
            value = self._get_nested(self._base, key)

        # If still not found, return default value
        if value is None:
            return default

        # Type conversion
        if value_type is not None:
            if isinstance(value, value_type):
                return value
            try:
                if value_type == bool:
                    # Special handling for bool
                    if isinstance(value, str):
                        return value.lower() in ('true', '1', 'yes')
                    return bool(value)
                return value_type(value)
            except (ValueError, TypeError, AttributeError):
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value

        Args:
            key: Configuration key, supports dot-separated path
            value: Configuration value
        """
        # Set to base configuration
        keys = key.split('.')
        target = self._base

        for k in keys[:-1]:
            if isinstance(target, dict):
                if k not in target:
                    target[k] = {}
                target = target[k]
            elif hasattr(target, k):
                # dataclass, get attribute
                target = getattr(target, k)
            else:
                # Create new dictionary
                new_target = {}
                setattr(target, k, new_target)
                target = new_target

        # Set final value
        if isinstance(target, dict):
            target[keys[-1]] = value
        else:
            setattr(target, keys[-1], value)

        logger.debug(f"Configuration updated: {key} = {value}")

    def get_section(self, section: str) -> Any:
        """
        Get configuration section

        Args:
            section: Section name, e.g. "separators", "audio_utils"

        Returns:
            Configuration section object
        """
        # Supports dataclass object
        if hasattr(self._base, section):
            return getattr(self._base, section)
        if isinstance(self._base, dict) and section in self._base:
            return self._base[section]
        raise ConfigError(f"Configuration section does not exist: {section}")

    def reset(self) -> None:
        """Reset to default configuration"""
        self._base = copy.deepcopy(DEFAULT_CONFIG)
        self._user = {}
        logger.info("Configuration has been reset to default values")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        # Supports dataclass object
        if hasattr(self._base, 'to_dict') and callable(self._base.to_dict):
            return self._base.to_dict()
        if isinstance(self._base, dict):
            return copy.deepcopy(self._base)
        return _safe_asdict(self._base)

    def validate(self) -> bool:
        """
        Validate configuration effectiveness

        Returns:
            Whether valid
        """
        errors = []

        # Validate device configuration
        device = self.get("separators.device")
        if device not in ["auto", "cpu", "cuda", "mps"]:
            errors.append(f"Invalid device type: {device}")

        # Validate numeric ranges
        for key, min_val, max_val in [
            ("audio_utils.max_file_size_mb", 1, 10000),
            ("audio_utils.max_duration_seconds", 1, 86400),
            ("security.max_path_depth", 1, 100),
        ]:
            value = self.get(key)
            if value is not None and not (min_val <= value <= max_val):
                errors.append(f"{key} value {value} out of range [{min_val}, {max_val}]")

        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            return False

        return True

    @staticmethod
    def _get_nested(obj: Any, key: str) -> Optional[Any]:
        """Get nested attribute, only supports '.' as separator"""
        # Only use '.' separator, do not replace '_'
        keys = key.split('.')
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k)
            elif hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                return None
            if obj is None:
                return None
        return obj


@lru_cache(maxsize=1)
def get_config_path() -> Path:
    """
    Get configuration path

    Priority:
    1. Environment variable SOMA_CONFIG_PATH
    2. ~/.soma/config.json
    """
    # Environment variable takes priority
    env_path = os.environ.get("SOMA_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser()

    # Default path
    default_path = Path("~/.soma/config.json").expanduser()
    return default_path


def get_config(
    path: Optional[Union[str, Path]] = None,
    auto_create: bool = True
) -> Config:
    """
    Get global configuration instance

    Args:
        path: Configuration path, defaults to get_config_path()
        auto_create: Whether to auto-create

    Returns:
        Config instance
    """
    if path is None:
        path = get_config_path()
    return Config.load(path, auto_create=auto_create)


# Initialize configuration module
def init_config() -> Config:
    """Initialize and return global configuration"""
    config = get_config()

    # Ensure necessary directories exist
    app_dir = Path(config.get("soma.app_dir", "~/.soma")).expanduser()
    for subdir in ["models", "cache", "temp", "logs", "workspace"]:
        (app_dir / subdir).mkdir(parents=True, exist_ok=True)

    return config
