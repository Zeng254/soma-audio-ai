"""
SOMA 统一异常模块

定义所有 SOMA 模块的异常基类，提供统一的异常体系。
"""


class SOMAError(Exception):
    """
    SOMA 异常基类

    所有 SOMA 模块抛出的异常都应继承此类。
    提供异常分类和统一处理接口。

    子类分类：
    - ConfigError: 配置相关错误
    - SecurityError: 安全相关错误
    - ModelError: 模型相关错误
    - AudioError: 音频处理错误
    - PipelineError: 流水线处理错误
    """

    # 错误类别
    CATEGORY = "general"

    def __init__(self, message: str = "", **kwargs):
        self.message = message
        self.extra = kwargs
        super().__init__(self.message)

    def __str__(self):
        if self.extra:
            extra_str = ", ".join(f"{k}={v!r}" for k, v in self.extra.items())
            return f"{self.__class__.__name__}({self.message!r}, {extra_str})"
        return f"{self.__class__.__name__}({self.message!r})"

    @property
    def category(self) -> str:
        """获取错误类别"""
        return self.CATEGORY

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "type": self.__class__.__name__,
            "category": self.category,
            "message": self.message,
            "extra": self.extra,
        }


# ============================================================================
# 配置相关错误
# ============================================================================


class ConfigError(SOMAError):
    """配置相关错误"""
    CATEGORY = "config"


class ConfigLoadError(ConfigError):
    """配置加载失败"""
    pass


class ConfigValidationError(ConfigError):
    """配置验证失败"""
    pass


class ConfigTypeError(ConfigError):
    """配置类型错误"""
    pass


# ============================================================================
# 安全相关错误
# ============================================================================


class SecurityError(SOMAError):
    """安全相关错误"""
    CATEGORY = "security"


class PathTraversalError(SecurityError):
    """路径遍历攻击检测"""
    pass


class AudioValidationError(SecurityError):
    """音频文件验证失败"""
    pass


class ModelSecurityError(SecurityError):
    """模型安全相关错误"""
    pass


class ModelVerificationError(ModelSecurityError):
    """模型验证失败"""
    pass


# ============================================================================
# 模型相关错误
# ============================================================================


class ModelError(SOMAError):
    """模型相关错误"""
    CATEGORY = "model"


class ModelLoadError(ModelError):
    """模型加载失败"""
    pass


class ModelNotFoundError(ModelError):
    """模型文件未找到"""
    pass


class ModelUnsupportedError(ModelError):
    """不支持的模型类型"""
    pass


class ModelDependencyError(ModelError):
    """模型依赖缺失"""
    pass


# ============================================================================
# 音频处理错误
# ============================================================================


class AudioError(SOMAError):
    """音频处理错误"""
    CATEGORY = "audio"


class AudioLoadError(AudioError):
    """音频加载失败"""
    pass


class AudioSaveError(AudioError):
    """音频保存失败"""
    pass


class AudioFormatError(AudioError):
    """不支持的音频格式"""
    pass


class AudioParameterError(AudioError):
    """音频参数错误"""
    pass


class AudioProcessingError(AudioError):
    """音频处理失败"""
    pass


# ============================================================================
# 分离器错误
# ============================================================================


class SeparatorError(SOMAError):
    """音频分离错误"""
    CATEGORY = "separator"


class SeparatorLoadError(SeparatorError):
    """分离器加载失败"""
    pass


class SeparationError(SeparatorError):
    """分离处理失败"""
    pass


# ============================================================================
# 声音转换错误
# ============================================================================


class VoiceConverterError(SOMAError):
    """声音转换错误"""
    CATEGORY = "voice_converter"


class VoiceConverterLoadError(VoiceConverterError):
    """转换器加载失败"""
    pass


class VoiceConversionError(VoiceConverterError):
    """转换处理失败"""
    pass


# ============================================================================
# 音效处理错误
# ============================================================================


class EffectError(SOMAError):
    """音效处理错误"""
    CATEGORY = "effect"


class EffectParameterError(EffectError):
    """音效参数错误"""
    pass


class EffectProcessingError(EffectError):
    """音效处理失败"""
    pass


# ============================================================================
# 流水线错误
# ============================================================================


class PipelineError(SOMAError):
    """流水线处理错误"""
    CATEGORY = "pipeline"


class PipelineNodeError(PipelineError):
    """流水线节点错误"""
    pass


class PipelineExecutionError(PipelineError):
    """流水线执行错误"""
    pass


# ============================================================================
# 工具函数
# ============================================================================


def is_soma_error(exc: Exception) -> bool:
    """判断是否为 SOMA 异常"""
    return isinstance(exc, SOMAError)


def get_error_category(exc: Exception) -> str:
    """获取异常类别"""
    if isinstance(exc, SOMAError):
        return exc.category
    return "unknown"


def format_error(exc: Exception) -> str:
    """格式化异常信息"""
    if isinstance(exc, SOMAError):
        return f"[{exc.category.upper()}] {exc.message}"
    return str(exc)
