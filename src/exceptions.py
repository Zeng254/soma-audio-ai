"""
SOMA Audio AI Unified Exception Hierarchy

Provides consistent exception hierarchy for error handling and debugging.
"""

from typing import Any, Optional


class SOMAError(Exception):
    """
    SOMA Base exception class
    
    Base class for all SOMA-related exceptions.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class SOMADependencyError(SOMAError, ImportError):
    """
    Dependency missing exception
    
    Raised when a required Python package is not installed.
    """

    def __init__(
        self,
        package: str,
        message: Optional[str] = None,
        install_hint: Optional[str] = None,
    ):
        self.package = package
        self.install_hint = install_hint or f"uv add {package}"
        msg = message or (
            f"Required package '{package}' is not installed.\n"
            f"Please install it with: {self.install_hint}"
        )
        super().__init__(msg, {"package": package, "install_hint": self.install_hint})


class SOMAModelError(SOMAError, ValueError):
    """
    Model error exception
    
    Raised when a model file does not exist, is corrupted, or cannot be loaded.
    """

    def __init__(
        self,
        model_path: str,
        message: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        self.model_path = model_path
        self.reason = reason
        msg = message or f"Failed to load model from '{model_path}'"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg, {"model_path": model_path, "reason": reason})


class SOMAModelNotFoundError(SOMAModelError):
    """
    Model file not found exception
    """

    def __init__(self, model_path: str, search_paths: Optional[list] = None):
        self.search_paths = search_paths or []
        msg = (
            f"Model file not found: '{model_path}'\n"
            f"Search paths: {search_paths or []}"
        )
        super().__init__(
            model_path,
            message=msg,
            reason="file not found",
        )


class SOMAModelCorruptedError(SOMAModelError):
    """
    Model file corrupted exception
    """

    def __init__(self, model_path: str, details: Optional[str] = None):
        super().__init__(
            model_path,
            message=f"Model file is corrupted: '{model_path}'",
            reason=f"corrupted{': ' + details if details else ''}",
        )


class SOMAValidationError(SOMAError, ValueError):
    """
    Parameter validation exception
    
    Raised when an input parameter is invalid.
    """

    def __init__(
        self,
        param_name: str,
        value: Any,
        message: Optional[str] = None,
        constraints: Optional[dict] = None,
    ):
        self.param_name = param_name
        self.value = value
        self.constraints = constraints or {}
        msg = message or f"Invalid value for parameter '{param_name}': {value}"
        if constraints:
            msg = f"{msg} | Constraints: {constraints}"
        super().__init__(
            msg,
            {"param_name": param_name, "value": value, "constraints": self.constraints},
        )


class SOMAConversionError(SOMAError):
    """
    Conversion failure exception
    
    Raised when audio conversion process fails.
    """

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.stage = stage
        self.original_error = original_error
        details = {"stage": stage}
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        super().__init__(message, details)


class SOMAAudioError(SOMAError):
    """
    Audio processing exception
    
    Raised when audio file reading or processing fails.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ):
        self.file_path = file_path
        self.sample_rate = sample_rate
        super().__init__(
            message,
            {"file_path": file_path, "sample_rate": sample_rate},
        )


class SOMARuntimeError(SOMAError, RuntimeError):
    """
    Runtime error exception
    
    Raised when runtime status is abnormal.
    """

    pass


class ConfigError(SOMAError, ValueError):
    """
    Configuration error exception
    
    Raised when configuration parameters are invalid or configuration file is corrupted.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[any] = None,
    ):
        self.config_key = config_key
        self.config_value = config_value
        details = {}
        if config_key is not None:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = config_value
        super().__init__(message, details)


class ConfigLoadError(ConfigError):
    """
    Configuration load error exception
    
    Raised when configuration file cannot be loaded or parsed.
    """

    pass


class ConfigValidationError(ConfigError):
    """
    Configuration validation error exception
    
    Raised when configuration validation fails.
    """

    pass


class ConfigTypeError(ConfigError):
    """
    Configuration type error exception
    
    Raised when configuration type conversion fails.
    """

    pass


class SecurityError(SOMAError):
    """
    Security error exception
    
    Raised when security validation fails.
    """

    pass


class PathTraversalError(SecurityError):
    """
    Path traversal error exception
    
    Raised when path traversal is detected.
    """

    def __init__(self, message: str = "Path traversal attack detected",
                 attempted_path: str = "", allowed_base: str = None, **kwargs):
        details = {"attempted_path": attempted_path, "allowed_base": allowed_base}
        super().__init__(message, details=details)


class AudioValidationError(SOMAValidationError):
    """
    Audio validation error exception
    
    Raised when audio validation fails.
    """

    pass


class ModelSecurityError(SecurityError):
    """
    Model security error exception
    
    Raised when model security validation fails.
    """

    pass


# ============================================================================
# Additional Exception Aliases for Backward Compatibility
# ============================================================================

class ModelError(SOMAModelError):
    """Alias for SOMAModelError for backward compatibility."""
    pass


class ModelLoadError(SOMAModelError):
    """Raised when model loading fails."""
    pass


class ModelNotFoundError(SOMAModelNotFoundError):
    """Raised when model file is not found."""
    pass


class AudioError(SOMAAudioError):
    """Alias for SOMAAudioError for backward compatibility."""
    pass


class AudioLoadError(SOMAAudioError):
    """Raised when audio loading fails."""
    pass


class AudioFormatError(SOMAAudioError):
    """Raised when audio format is invalid."""
    pass


class AudioProcessingError(SOMAAudioError):
    """Raised when audio processing fails."""
    pass


class SeparatorError(SOMAError):
    """Base exception for separator-related errors."""
    pass


class SeparationError(SeparatorError):
    """Raised when audio separation fails."""
    pass


class VoiceConverterError(SOMAError):
    """Base exception for voice converter-related errors."""
    pass


class VoiceConversionError(VoiceConverterError):
    """Raised when voice conversion fails."""
    pass


# Timeout configuration constants
TIMEOUT_MINUTES = 15  # minutes
TIMEOUT_SECONDS = TIMEOUT_MINUTES * 60  # seconds

# Task tracking
RUNNING_TASKS = {}  # Tracks currently running tasks

# Error classifier
def classify_error(error: Exception) -> str:
    """Classify error type for logging and handling."""
    if isinstance(error, SOMADependencyError):
        return "dependency"
    elif isinstance(error, SOMAModelError):
        return "model"
    elif isinstance(error, SOMAValidationError):
        return "validation"
    elif isinstance(error, SOMAConversionError):
        return "conversion"
    elif isinstance(error, SOMAAudioError):
        return "audio"
    elif isinstance(error, ConfigError):
        return "config"
    else:
        return "unknown"


def is_soma_error(error: Exception) -> bool:
    """Check if an error is a SOMA-related error."""
    return isinstance(error, SOMAError)


def get_error_category(error: Exception) -> str:
    """Get the category of an error.
    
    Args:
        error: The error to categorize.
        
    Returns:
        The category of the error (e.g., 'model', 'validation', 'conversion', 'audio', 'config', 'unknown').
    """
    return classify_error(error)


def format_error(error: Exception, include_traceback: bool = False) -> str:
    """Format an error message.
    
    Args:
        error: The error to format.
        include_traceback: Whether to include the traceback.
        
    Returns:
        A formatted error message string.
    """
    import traceback
    message = f"{type(error).__name__}: {error}"
    if include_traceback:
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        message += "\n" + "".join(tb)
    return message
