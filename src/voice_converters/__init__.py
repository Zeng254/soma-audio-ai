"""
SOMA Voice Converters Module
声音转换模块 - 双引擎架构

提供统一的声音转换接口，支持:
- RVC v2 (Retrieval-Based Voice Conversion)
- So-VITS-SVC 4.1

模块结构:
- base: 抽象基类和通用接口
- rvc_converter: RVC v2 引擎实现
- sovits_converter: So-VITS-SVC 引擎实现
- factory: 引擎工厂和自动识别
"""

from .base import (
    # 基类和枚举
    BaseVoiceConverter,
    ConversionParams,
    ConversionResult,
    ModelInfo,
    ConverterType,
    F0Method,
    LazyImportMixin,
    EngineCapability,
)

# 便捷访问列表
__all_base__ = [
    "BaseVoiceConverter",
    "ConversionParams",
    "ConversionResult",
    "ModelInfo",
    "ConverterType",
    "F0Method",
    "LazyImportMixin",
    "EngineCapability",
]

# 延迟导入引擎类
def __getattr__(name: str):
    """延迟导入引擎类"""
    if name == "RVCConverter":
        from .rvc_converter import RVCConverter
        return RVCConverter
    
    if name == "SoVITSConverter":
        from .sovits_converter import SoVITSConverter
        return SoVITSConverter
    
    if name == "ConverterFactory":
        from .factory import ConverterFactory
        return ConverterFactory
    
    if name == "VoiceConverterManager":
        from .factory import VoiceConverterManager
        return VoiceConverterManager
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 导出列表
__all__ = [
    # 基类和接口
    "BaseVoiceConverter",
    "ConversionParams",
    "ConversionResult",
    "ModelInfo",
    "ConverterType",
    "F0Method",
    "LazyImportMixin",
    "EngineCapability",
    
    # 引擎类
    "RVCConverter",
    "SoVITSConverter",
    
    # 工厂和管理器
    "ConverterFactory",
    "VoiceConverterManager",
]


# 便捷函数
def create_converter(
    model_path: str,
    config_path: str = None,
    index_path: str = None,
    engine: str = None,
    device: str = None,
    **kwargs
):
    """
    创建声音转换器 (便捷函数)
    
    自动识别模型类型并创建转换器。
    
    Args:
        model_path: 模型文件路径
        config_path: 配置文件路径
        index_path: 索引文件路径
        engine: 强制指定引擎 ('rvc', 'sovits')
        device: 运行设备
        **kwargs: 其他参数
        
    Returns:
        BaseVoiceConverter: 转换器实例
        
    Example:
        >>> converter = create_converter("path/to/model.pth")
        >>> result = converter.convert(audio, sample_rate)
    """
    from .factory import ConverterFactory
    
    return ConverterFactory.create_converter(
        model_path=model_path,
        config_path=config_path,
        index_path=index_path,
        engine=engine,
        device=device,
        **kwargs
    )


def get_available_engines():
    """
    获取可用的引擎列表
    
    Returns:
        List[Dict]: 引擎信息列表
        
    Example:
        >>> engines = get_available_engines()
        >>> for engine in engines:
        ...     print(f"{engine['name']}: {engine['available']}")
    """
    from .factory import ConverterFactory
    
    return ConverterFactory.get_available_engines()


def get_conversion_params(engine: str = None) -> "ConversionParams":
    """
    获取转换参数
    
    Args:
        engine: 引擎类型
        
    Returns:
        ConversionParams: 推荐参数
    """
    from .base import ConversionParams
    from .factory import ConverterFactory
    
    if engine is None:
        return ConversionParams()
    
    return ConverterFactory.get_recommended_params(engine)


# 版本信息
__version__ = "0.1.0"
