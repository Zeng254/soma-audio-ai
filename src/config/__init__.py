"""
SOMA 配置中心

提供统一的配置管理系统，支持层级覆盖和类型安全。

主要组件:
- Config: 主配置类
- get_config: 获取全局配置实例
- init_config: 初始化配置

层级覆盖顺序:
1. 默认值 (defaults.py)
2. 配置文件 (JSON/YAML)
3. 用户输入 (运行时覆盖)

使用示例:
    from src.config import get_config, Config

    # 获取全局配置
    config = get_config()

    # 获取配置值
    sample_rate = config.get("audio_utils.default_sample_rate")
    device = config.get("separators.device", default="cuda")

    # 设置配置值
    config.set("separators.device", "cuda")

    # 保存配置
    config.save()

    # 直接导入默认值
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
    # 配置类
    "Config",

    # 异常
    "ConfigError",
    "ConfigLoadError",
    "ConfigValidationError",
    "ConfigTypeError",

    # 默认配置
    "DEFAULT_CONFIG",
    "SomaDefaults",
    "SeparatorDefaults",
    "VoiceConverterDefaults",
    "EffectsDefaults",
    "ConverterDefaults",
    "AudioUtilsDefaults",
    "SecurityDefaults",
    "LoggingDefaults",

    # 工厂函数
    "get_config",
    "get_config_path",
    "init_config",
]
