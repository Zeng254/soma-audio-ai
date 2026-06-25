"""
Voice Converter Factory - 声音转换引擎工厂

提供统一的引擎创建和管理接口。
支持自动识别模型类型。
"""

from typing import Optional, Dict, List, Type, Any, Union
from pathlib import Path
import json
import os

from .base import (
    BaseVoiceConverter,
    ConverterType,
    ConversionParams,
    ModelInfo,
)


class ConverterFactory:
    """
    声音转换器工厂
    
    负责:
    - 自动识别模型类型
    - 创建合适的转换器实例
    - 管理转换器生命周期
    - 缓存已创建的转换器
    """
    
    # 引擎注册表
    _engines: Dict[ConverterType, Type[BaseVoiceConverter]] = {}
    
    # 模型类型标识
    MODEL_TYPE_INDICATORS = {
        # RVC 标识
        ".pth": "rvc",
        "rvc": ["model", "emb", "f0"],
        
        # SoVITS 标识
        "G_*.pth": "sovits",
        "sovits": ["config", "mel", "spk"],
    }
    
    @classmethod
    def register_engine(
        cls,
        converter_type: ConverterType,
        engine_class: Type[BaseVoiceConverter]
    ):
        """
        注册声音转换引擎
        
        Args:
            converter_type: 引擎类型
            engine_class: 引擎类
        """
        cls._engines[converter_type] = engine_class
    
    @classmethod
    def create_converter(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
        index_path: Optional[str] = None,
        engine: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseVoiceConverter:
        """
        创建声音转换器
        
        自动识别模型类型并创建对应的转换器。
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
            index_path: 索引文件路径
            engine: 强制指定引擎 ('rvc', 'sovits')
            device: 运行设备
            **kwargs: 其他参数
            
        Returns:
            BaseVoiceConverter: 转换器实例
            
        Raises:
            FileNotFoundError: 模型文件不存在
            ValueError: 无法识别模型类型
        """
        model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 如果没有提供 config_path，尝试查找
        if config_path is None:
            config_path = cls._find_config(model_file)
        
        # 识别模型类型
        if engine is None:
            engine = cls.identify_model_type(
                model_path,
                config_path,
                index_path
            )
        
        # 获取引擎类型
        try:
            converter_type = ConverterType(engine.lower())
        except ValueError:
            raise ValueError(
                f"Unknown engine type: {engine}. "
                f"Supported: {', '.join([e.value for e in ConverterType])}"
            )
        
        # 创建转换器
        converter = cls._create_converter_instance(
            converter_type,
            device,
            **kwargs
        )
        
        # 加载模型
        converter.load_model(
            model_path,
            config_path=config_path,
            index_path=index_path,
            **kwargs
        )
        
        return converter
    
    @classmethod
    def _create_converter_instance(
        cls,
        converter_type: ConverterType,
        device: Optional[str],
        **kwargs
    ) -> BaseVoiceConverter:
        """
        创建转换器实例
        
        Args:
            converter_type: 转换器类型
            device: 设备
            **kwargs: 其他参数
            
        Returns:
            BaseVoiceConverter: 转换器实例
        """
        # 检查是否已注册
        if converter_type in cls._engines:
            engine_class = cls._engines[converter_type]
        else:
            # 动态导入
            if converter_type == ConverterType.RVC:
                from .rvc_converter import RVCConverter
                engine_class = RVCConverter
            elif converter_type == ConverterType.SOVITS:
                from .sovits_converter import SoVITSConverter
                engine_class = SoVITSConverter
            else:
                raise ValueError(f"Unsupported converter type: {converter_type}")
        
        # 创建实例
        return engine_class(device=device, **kwargs)
    
    @classmethod
    def identify_model_type(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
        index_path: Optional[str] = None,
    ) -> str:
        """
        识别模型类型
        
        Args:
            model_path: 模型路径
            config_path: 配置文件路径
            index_path: 索引文件路径
            
        Returns:
            str: 模型类型 ('rvc' 或 'sovits')
        """
        model_file = Path(model_path)
        
        # 1. 从文件名判断
        model_name = model_file.name.lower()
        
        if model_name.startswith("G_") or model_name.startswith("D_"):
            # SoVITS 命名规范
            return "sovits"
        
        # 2. 从配置判断
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # SoVITS 配置特征
                if "train" in config or "model" in config:
                    if "spk" in config or "n_speakers" in config:
                        return "sovits"
                
                # RVC 配置特征
                if "emb" in config or "f0" in config:
                    return "rvc"
                    
            except Exception:
                pass
        
        # 3. 从索引文件判断
        if index_path and Path(index_path).exists():
            index_file = Path(index_path)
            if index_file.suffix == ".index":
                return "rvc"
        
        # 4. 检查同目录下的配置
        possible_configs = [
            model_file.parent / "config.json",
            model_file.parent / "configs" / "config.json",
            model_file.parent.parent / "config.json",
        ]
        
        for config_file in possible_configs:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    if "train" in config:
                        return "sovits"
                    elif "emb" in config or "f0" in config:
                        return "rvc"
                except Exception:
                    continue
        
        # 5. 默认值 (优先 RVC)
        return "rvc"
    
    @classmethod
    def _find_config(cls, model_file: Path) -> Optional[str]:
        """查找配置文件"""
        # SoVITS 配置查找路径
        sovits_paths = [
            model_file.parent / "config.json",
            model_file.parent / "configs" / "config.json",
            model_file.parent / "sovits_config.json",
            model_file.parent.parent / "config.json",
        ]
        
        for path in sovits_paths:
            if path.exists():
                return str(path)
        
        # RVC 不需要配置文件 (可选)
        return None
    
    @classmethod
    def get_available_engines(cls) -> List[Dict[str, Any]]:
        """
        获取可用的引擎列表
        
        Returns:
            List[Dict]: 引擎信息列表
        """
        engines = []
        
        # 检查 RVC
        try:
            from .rvc_converter import RVCConverter
            if RVCConverter.is_available():
                engines.append({
                    "type": "rvc",
                    "name": RVCConverter.get_engine_name(),
                    "supported_formats": RVCConverter.get_supported_formats(),
                    "available": True,
                })
        except ImportError:
            engines.append({
                "type": "rvc",
                "name": "RVC v2",
                "available": False,
                "error": "torch not installed",
            })
        
        # 检查 SoVITS
        try:
            from .sovits_converter import SoVITSConverter
            if SoVITSConverter.is_available():
                engines.append({
                    "type": "sovits",
                    "name": SoVITSConverter.get_engine_name(),
                    "supported_formats": SoVITSConverter.get_supported_formats(),
                    "available": True,
                })
        except ImportError:
            engines.append({
                "type": "sovits",
                "name": "So-VITS-SVC 4.1",
                "available": False,
                "error": "torch not installed",
            })
        
        return engines
    
    @classmethod
    def get_recommended_params(cls, engine: str) -> ConversionParams:
        """
        获取引擎推荐参数
        
        Args:
            engine: 引擎类型
            
        Returns:
            ConversionParams: 推荐参数
        """
        if engine == "rvc":
            return ConversionParams(
                pitch_shift=0,
                pitch_algo="rmvpe",
                vpm=0.5,
                timbre_protection=0.5,
                rms_mix=0.5,
            )
        elif engine == "sovits":
            return ConversionParams(
                pitch_shift=0,
                pitch_algo="dio",
                vpm=0.5,
                timbre_protection=0.5,
                rms_mix=0.5,
            )
        else:
            return ConversionParams()


class VoiceConverterManager:
    """
    声音转换器管理器
    
    生命周期管理和资源共享
    """
    
    def __init__(self):
        """初始化管理器"""
        self._converters: Dict[str, BaseVoiceConverter] = {}
        self._current: Optional[str] = None
    
    def load(
        self,
        name: str,
        model_path: str,
        **kwargs
    ) -> BaseVoiceConverter:
        """
        加载转换器
        
        Args:
            name: 转换器名称
            model_path: 模型路径
            **kwargs: 其他参数
            
        Returns:
            BaseVoiceConverter: 转换器实例
        """
        # 如果已存在，先卸载
        if name in self._converters:
            self.unload(name)
        
        # 创建新的转换器
        converter = ConverterFactory.create_converter(
            model_path,
            **kwargs
        )
        
        self._converters[name] = converter
        self._current = name
        
        return converter
    
    def get(self, name: Optional[str] = None) -> Optional[BaseVoiceConverter]:
        """
        获取转换器
        
        Args:
            name: 转换器名称，None 则返回当前
            
        Returns:
            BaseVoiceConverter 或 None
        """
        if name is None:
            name = self._current
        
        return self._converters.get(name)
    
    def unload(self, name: str):
        """
        卸载转换器
        
        Args:
            name: 转换器名称
        """
        if name in self._converters:
            converter = self._converters[name]
            converter.unload()
            del self._converters[name]
            
            if self._current == name:
                # 选择另一个作为当前
                self._current = next(iter(self._converters.keys()), None)
    
    def unload_all(self):
        """卸载所有转换器"""
        for converter in self._converters.values():
            converter.unload()
        self._converters.clear()
        self._current = None
    
    def list_converters(self) -> List[str]:
        """列出已加载的转换器"""
        return list(self._converters.keys())
    
    @property
    def current(self) -> Optional[BaseVoiceConverter]:
        """获取当前转换器"""
        return self.get()
    
    def __enter__(self) -> "VoiceConverterManager":
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.unload_all()
        return False


# 注册默认引擎（惰性导入，失败时记录日志但不崩溃）
import logging

logger = logging.getLogger(__name__)

try:
    from .rvc_converter import RVCConverter
    from .sovits_converter import SoVITSConverter
    
    ConverterFactory.register_engine(ConverterType.RVC, RVCConverter)
    ConverterFactory.register_engine(ConverterType.SOVITS, SoVITSConverter)
except ImportError as e:
    logger.warning(
        f"Failed to register default voice converter engines: {e}. "
        "Some features may not be available."
    )
