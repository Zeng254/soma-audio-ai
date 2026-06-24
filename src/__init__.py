"""
SOMA - Smart Omnimedia Audio

AI 驱动的音频处理工作站，集成声音分离、音效处理、音频生成等功能。

主要模块:
- separators: 音频分离器（人声/伴奏分离）
- effects: 音效处理器（均衡器、混响、音调变换）
- converters: 格式转换器
- voice_converters: 声音转换器（双引擎架构）
- pipeline: 处理流水线
- config: 配置中心
- security: 安全验证
- utils: 通用工具

使用示例:
    from src import Config, AudioLoader

    # 加载配置
    config = Config.load("~/.soma/config.json")

    # 加载音频
    loader = AudioLoader()
    audio, sr = loader.load("input.wav")
"""

# 异常导出
from src.exceptions import (
    # 基类
    SOMAError,

    # 配置异常
    ConfigError,
    ConfigLoadError,
    ConfigValidationError,
    ConfigTypeError,

    # 安全异常
    SecurityError,
    PathTraversalError,
    AudioValidationError,
    ModelSecurityError,

    # 模型异常
    ModelError,
    ModelLoadError,
    ModelNotFoundError,

    # 音频异常
    AudioError,
    AudioLoadError,
    AudioFormatError,

    # 分离器异常
    SeparatorError,
    SeparationError,

    # 声音转换异常
    VoiceConverterError,
    VoiceConversionError,

    # 工具函数
    is_soma_error,
    get_error_category,
    format_error,
)

__all__ = [
    # 版本信息
    "__version__",

    # 异常
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
    "is_soma_error",
    "get_error_category",
    "format_error",
]
