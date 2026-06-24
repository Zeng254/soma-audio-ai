"""
SOMA 安全模块

提供输入验证和安全检查功能，包括：
- 路径安全校验 (PathValidator)
- 音频文件验证 (AudioValidator)
- 安全模型加载 (SafeModelLoader)

层级结构:
    src/security/
    ├── __init__.py         # 模块导出
    ├── path_validator.py    # 路径安全校验
    ├── audio_validator.py   # 音频文件验证
    └── model_loader.py      # 安全模型加载

使用示例:
    # 路径验证
    from src.security import PathValidator, safe_path

    validator = PathValidator()
    safe = validator.validate("/path/to/file")

    # 音频验证
    from src.security import AudioValidator, validate_audio

    result = validate_audio("/path/to/audio.wav")
    if result.is_valid:
        print(f"采样率: {result.metadata.sample_rate}")

    # 模型加载
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
    AudioValidator,
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
    # 路径验证
    "PathValidator",
    "PathTraversalError",
    "SecurityError",
    "safe_path",
    "safe_join",
    "ensure_directory",
    "get_validator",

    # 音频验证
    "AudioValidator",
    "AudioFormat",
    "AudioMetadata",
    "AudioValidationResult",
    "AudioValidationError",
    "AudioFormatError",
    "validate_audio",
    "get_audio_validator",

    # 模型加载
    "SafeModelLoader",
    "ModelMetadata",
    "ModelLoadError",
    "ModelVerificationError",
    "load_model",
    "get_model_loader",
]
