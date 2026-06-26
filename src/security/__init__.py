"""
SOMA Security module

Provides input validation and security checks，Including：
- Path security validation (PathValidator)
- AudioFile validation (SecureAudioValidator)
- SecurityModel loading (SafeModelLoader)

Layered structure:
    src/security/
    ├── __init__.py         # ModuleExport
    ├── path_validator.py    # Path security validation
    ├── audio_validator.py   # AudioFile validation
    └── model_loader.py      # SecurityModel loading

Usage example:
    # PathValidate
    from src.security import PathValidator, safe_path

    validator = PathValidator()
    safe = validator.validate("/path/to/file")

    # AudioValidate
    from src.security import SecureAudioValidator, validate_audio

    result = validate_audio("/path/to/audio.wav")
    if result.is_valid:
        print(f"Sample rate: {result.metadata.sample_rate}")

    # Model loading
    from src.security import SafeModelLoader, load_model

    model = load_model("path/to/model.pth")
"""

from .path_validator import (
    PathValidator,
    PathTraversalError,
    SecurityError,
    safe_path,
    safe_join,
    ensure_directory,
    get_validator,
)

from .audio_validator import (
    SecureAudioValidator,
    AudioFormat,
    AudioMetadata,
    AudioValidationResult,
    AudioValidationError,
    AudioFormatError,
    validate_audio,
    get_audio_validator,
)

from .model_loader import (
    SafeModelLoader,
    ModelMetadata,
    ModelLoadError,
    ModelVerificationError,
    load_model,
    get_model_loader,
)

__all__ = [
    # PathValidate
    "PathValidator",
    "PathTraversalError",
    "SecurityError",
    "safe_path",
    "safe_join",
    "ensure_directory",
    "get_validator",

    # AudioValidate
    "SecureAudioValidator",
    "AudioFormat",
    "AudioMetadata",
    "AudioValidationResult",
    "AudioValidationError",
    "AudioFormatError",
    "validate_audio",
    "get_audio_validator",

    # Model loading
    "SafeModelLoader",
    "ModelMetadata",
    "ModelLoadError",
    "ModelVerificationError",
    "load_model",
    "get_model_loader",
]
