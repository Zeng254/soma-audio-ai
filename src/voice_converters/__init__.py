"""
SOMA Voice Converters Module
Voice conversion module - Dual engine architecture

Provides unified voice conversion interface, supports:
- RVC v2 (Retrieval-Based Voice Conversion)
- So-VITS-SVC 4.1

Module structure:
- base: Abstract base class and common interface
- rvc_converter: RVC v2 engine implementation
- sovits_converter: So-VITS-SVC engine implementation
- factory: Engine factory and automatic detection
"""

from .base import (
    # Base class and enum
    BaseVoiceConverter,
    ConversionParams,
    ConversionResult,
    ModelInfo,
    ConverterType,
    F0Method,
    LazyImportMixin,
    EngineCapability,
)

# Convenience access list
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

# Delay import engine class
def __getattr__(name: str):
    """Delay import engine class"""
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


# ExportList
__all__ = [
    # Base class and interface
    "BaseVoiceConverter",
    "ConversionParams",
    "ConversionResult",
    "ModelInfo",
    "ConverterType",
    "F0Method",
    "LazyImportMixin",
    "EngineCapability",
    
    # Engine class
    "RVCConverter",
    "SoVITSConverter",
    
    # Factory and manager
    "ConverterFactory",
    "VoiceConverterManager",
]


# Convenience function
def create_converter(
    model_path: str,
    config_path: str = None,
    index_path: str = None,
    engine: str = None,
    device: str = None,
    **kwargs
):
    """
    Create voice converter (convenience function)
    
    Automatically detect model type and create converter.
    
    Args:
        model_path: ModelFile path
        config_path: Configuration filePath
        index_path: IndexFile path
        engine: Force specify engine ('rvc', 'sovits')
        device: Run device
        **kwargs: OtherParameter
        
    Returns:
        BaseVoiceConverter: Converter instance
        
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
    Get available engine list
    
    Returns:
        List[Dict]: Engine info list
        
    Example:
        >>> engines = get_available_engines()
        >>> for engine in engines:
        ...     print(f"{engine['name']}: {engine['available']}")
    """
    from .factory import ConverterFactory
    
    return ConverterFactory.get_available_engines()


def get_conversion_params(engine: str = None) -> "ConversionParams":
    """
    GetConvertParameter
    
    Args:
        engine: Engine class type
        
    Returns:
        ConversionParams: Recommended parameters
    """
    from .base import ConversionParams
    from .factory import ConverterFactory
    
    if engine is None:
        return ConversionParams()
    
    return ConverterFactory.get_recommended_params(engine)


# Version info
__version__ = "0.1.0"
