"""
SOMA Audio AI 统一异常体系

提供一致的异常层次结构，方便错误处理和调试。
"""

from typing import Optional


class SOMAError(Exception):
    """
    SOMA 基础异常类
    
    所有 SOMA 相关异常的基类。
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
    依赖缺失异常
    
    当必需的 Python 包未安装时抛出。
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
    模型错误异常
    
    当模型文件不存在、损坏或无法加载时抛出。
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
    模型文件未找到异常
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
    模型文件损坏异常
    """

    def __init__(self, model_path: str, details: Optional[str] = None):
        super().__init__(
            model_path,
            message=f"Model file is corrupted: '{model_path}'",
            reason=f"corrupted{': ' + details if details else ''}",
        )


class SOMAValidationError(SOMAError, ValueError):
    """
    参数校验异常
    
    当输入参数无效时抛出。
    """

    def __init__(
        self,
        param_name: str,
        value: any,
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
    转换失败异常
    
    当音频转换过程失败时抛出。
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
    音频处理异常
    
    当音频文件读取或处理失败时抛出。
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
    运行时错误异常
    
    当运行时状态异常时抛出。
    """

    pass


class ConfigError(SOMAError, ValueError):
    """
    配置错误异常
    
    当配置参数无效或配置文件损坏时抛出。
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
    配置加载错误异常
    
    当配置文件无法加载或解析时抛出。
    """

    def __init__(
        self,
        config_path: str,
        reason: Optional[str] = None,
    ):
        self.config_path = config_path
        self.reason = reason
        msg = f"Failed to load config from '{config_path}'"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg, config_key=config_path, config_value=reason)


class ConfigValidationError(ConfigError):
    """
    配置验证错误异常
    
    当配置值验证失败时抛出。
    """

    def __init__(
        self,
        config_key: str,
        value: any,
        constraints: Optional[dict] = None,
    ):
        self.constraints = constraints or {}
        msg = f"Config validation failed for '{config_key}': {value}"
        if constraints:
            msg = f"{msg} (expected: {constraints})"
        super().__init__(msg, config_key=config_key, config_value=value)
        self.details["constraints"] = self.constraints


class ConfigTypeError(ConfigError, TypeError):
    """
    配置类型错误异常
    
    当配置值类型错误时抛出。
    """

    def __init__(
        self,
        config_key: str,
        expected_type: type,
        actual_type: type,
    ):
        self.expected_type = expected_type
        self.actual_type = actual_type
        msg = (
            f"Config type error for '{config_key}': "
            f"expected {expected_type.__name__}, got {actual_type.__name__}"
        )
        super().__init__(msg, config_key=config_key, config_value=actual_type.__name__)
        self.details["expected_type"] = expected_type.__name__
        self.details["actual_type"] = actual_type.__name__


class SecurityError(SOMAError):
    """
    安全错误异常
    
    当安全检查失败时抛出，如路径遍历、恶意文件等。
    """

    def __init__(
        self,
        message: str,
        security_type: Optional[str] = None,
        path: Optional[str] = None,
    ):
        self.security_type = security_type
        self.path = path
        details = {}
        if security_type:
            details["security_type"] = security_type
        if path:
            details["path"] = path
        super().__init__(message, details)


class PathTraversalError(SecurityError):
    """
    路径遍历攻击异常
    
    当检测到路径遍历攻击时抛出。
    """

    def __init__(
        self,
        attempted_path: str,
        allowed_base: Optional[str] = None,
    ):
        self.attempted_path = attempted_path
        self.allowed_base = allowed_base
        msg = f"Path traversal attempt detected: '{attempted_path}'"
        if allowed_base:
            msg = f"{msg} (allowed base: '{allowed_base}')"
        super().__init__(msg, security_type="path_traversal", path=attempted_path)
        self.details["allowed_base"] = allowed_base


class AudioValidationError(SOMAError, ValueError):
    """
    音频验证错误异常
    
    当音频文件验证失败时抛出。
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        validation_type: Optional[str] = None,
    ):
        self.validation_type = validation_type
        super().__init__(
            message,
            {"file_path": file_path, "validation_type": validation_type},
        )


class ModelSecurityError(SecurityError):
    """
    模型安全错误异常
    
    当模型文件存在安全风险时抛出，如恶意代码、篡改等。
    """

    def __init__(
        self,
        model_path: str,
        security_issue: Optional[str] = None,
    ):
        self.security_issue = security_issue
        msg = f"Security issue detected in model file: '{model_path}'"
        if security_issue:
            msg = f"{msg}: {security_issue}"
        super().__init__(msg, security_type="model_security", path=model_path)
        self.details["security_issue"] = security_issue


# 别名：ModelError (简化版)
ModelError = SOMAModelError

# 别名：ModelLoadError (简化版)
ModelLoadError = SOMAModelNotFoundError

# 别名：ModelNotFoundError (简化版)
ModelNotFoundError = SOMAModelNotFoundError

# 别名：AudioError (简化版)
AudioError = SOMAAudioError

# 别名：AudioLoadError (简化版)
AudioLoadError = SOMAAudioError

# 别名：AudioFormatError (简化版)
AudioFormatError = SOMAAudioError

# 别名：AudioProcessingError (简化版)
AudioProcessingError = SOMAAudioError

# 别名：SeparatorError (简化版)
SeparatorError = SOMAConversionError

# 别名：SeparationError (简化版)
SeparationError = SOMAConversionError

# 别名：VoiceConverterError (简化版)
VoiceConverterError = SOMAConversionError

# 别名：VoiceConversionError (简化版)
VoiceConversionError = SOMAConversionError


# 辅助函数
def is_soma_error(exc: Exception) -> bool:
    """
    检查异常是否为 SOMA 异常或其子类
    
    Args:
        exc: 异常对象
        
    Returns:
        bool: 是否为 SOMA 异常
    """
    return isinstance(exc, SOMAError)


def get_error_category(exc: Exception) -> str:
    """
    获取异常类别
    
    Args:
        exc: 异常对象
        
    Returns:
        str: 异常类别 ('soma', 'config', 'model', 'audio', 'security', 'conversion', 'unknown')
    """
    if isinstance(exc, SOMAError):
        if isinstance(exc, ConfigError):
            return "config"
        elif isinstance(exc, SOMAModelError):
            return "model"
        elif isinstance(exc, SOMAAudioError):
            return "audio"
        elif isinstance(exc, SecurityError):
            return "security"
        elif isinstance(exc, SOMAConversionError):
            return "conversion"
        else:
            return "soma"
    return "unknown"


def format_error(exc: Exception, include_details: bool = True) -> str:
    """
    格式化异常信息
    
    Args:
        exc: 异常对象
        include_details: 是否包含详细信息
        
    Returns:
        str: 格式化的错误信息
    """
    if isinstance(exc, SOMAError):
        if include_details and exc.details:
            details_str = ", ".join(f"{k}={v}" for k, v in exc.details.items())
            return f"{exc.message} ({details_str})"
        return exc.message
    
    # 非 SOMA 异常
    if include_details:
        return f"{type(exc).__name__}: {str(exc)}"
    return str(exc)


# 导出所有异常类
__all__ = [
    "SOMAError",
    "SOMADependencyError",
    "SOMAModelError",
    "SOMAModelNotFoundError",
    "SOMAModelCorruptedError",
    "SOMAValidationError",
    "SOMAConversionError",
    "SOMAAudioError",
    "SOMARuntimeError",
    "ConfigError",
    "ConfigLoadError",
    "ConfigValidationError",
    "ConfigTypeError",
    "SecurityError",
    "PathTraversalError",
    "AudioValidationError",
    "ModelSecurityError",
    "ModelError",  # 别名
    "ModelLoadError",  # 别名
    "ModelNotFoundError",  # 别名
    "AudioError",  # 别名
    "AudioLoadError",  # 别名
    "AudioFormatError",  # 别名
    "AudioProcessingError",  # 别名
    "SeparatorError",  # 别名
    "SeparationError",  # 别名
    "VoiceConverterError",  # 别名
    "VoiceConversionError",  # 别名
    "is_soma_error",  # 辅助函数
    "get_error_category",  # 辅助函数
]
