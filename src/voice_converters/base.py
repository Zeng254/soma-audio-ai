"""
Base Voice Converter - 声音转换抽象基类
定义所有声音转换引擎的通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import numpy as np

from src.exceptions import (
    SOMAError,
    SOMADependencyError,
    SOMAValidationError,
    SOMAConversionError,
)


class F0Method(Enum):
    """基频提取方法"""
    PM = "pm"           # 倒谱法，快速但精度一般
    DIO = "dio"         # DIO 算法，精度较高
    CREPE = "crepe"     # 深度学习法，最精确但最慢
    CREPE_TINY = "crepe_tiny"  # 轻量级 CREPE
    HARVEST = "harvest"  # Harvest 算法，稳定但慢
    RMVPE = "rmvpe"     # 重采样基频预测


class ConverterType(Enum):
    """转换器类型"""
    RVC = "rvc"
    SOVITS = "sovits"
    UNKNOWN = "unknown"


@dataclass
class ConversionParams:
    """
    声音转换通用参数
    
    这些参数在所有引擎中统一支持，
    底层会自动映射到各引擎的具体参数
    """
    # 音高调整
    pitch_shift: float = 0.0          # 半音调整 (-24 to +24)
    pitch_algo: str = "rmvpe"         # 音高算法 (pm/dio/crepe/harvest/rmvpe)
    
    # 音色控制
    vpm: float = 0.5                  # 音素周期匹配 (0.0-1.0)
    timbre_protection: float = 0.5    # 音色保护 (0.0-1.0)
    
    # 响度控制
    rms_mix: float = 0.5              # RMS 响度混合 (0.0-1.0)
    loudness_match: bool = True       # 是否匹配响度
    
    # 质量控制
    sample_rate: int = 40000          # 输出采样率
    hop_length: int = 128              # 帧移
    f0_factor: float = 1.0            # 基频缩放因子
    
    # 扩散参数 (SoVITS)
    diffusion_steps: int = 10          # 扩散步数
    diffusion_temperature: float = 1.0 # 扩散温度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pitch_shift": self.pitch_shift,
            "pitch_algo": self.pitch_algo,
            "vpm": self.vpm,
            "timbre_protection": self.timbre_protection,
            "rms_mix": self.rms_mix,
            "loudness_match": self.loudness_match,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "f0_factor": self.f0_factor,
            "diffusion_steps": self.diffusion_steps,
            "diffusion_temperature": self.diffusion_temperature,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversionParams":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConversionResult:
    """
    声音转换结果
    
    统一返回格式，包含转换后的音频和元信息
    """
    audio: np.ndarray                 # 转换后的音频 (samples, channels) 或 (samples,)
    sampling_rate: int                # 采样率
    info: Dict[str, Any] = field(default_factory=dict)  # 转换信息
    
    # 质量指标
    pitch_range: Optional[tuple] = None  # (min_hz, max_hz)
    duration: Optional[float] = None     # 持续时间(秒)
    rms_db: Optional[float] = None       # RMS 电平(dB)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "audio_shape": self.audio.shape if self.audio is not None else None,
            "sampling_rate": self.sampling_rate,
            "info": self.info,
            "pitch_range": self.pitch_range,
            "duration": self.duration,
            "rms_db": self.rms_db,
        }


@dataclass
class ModelInfo:
    """模型信息"""
    name: str                          # 模型名称
    type: ConverterType                 # 模型类型
    version: Optional[str] = None      # 版本号
    sample_rate: int = 40000          # 原始采样率
    description: Optional[str] = None  # 描述
    file_path: Optional[str] = None    # 文件路径
    config_path: Optional[str] = None  # 配置文件路径
    index_path: Optional[str] = None   # 索引文件路径
    is_loaded: bool = False           # 是否已加载
    memory_usage: Optional[int] = None # 内存占用(字节)
    
    def __repr__(self) -> str:
        return f"ModelInfo({self.type.value}: {self.name})"


class BaseVoiceConverter(ABC):
    """
    声音转换器基类
    
    定义所有声音转换引擎的通用接口。
    支持按需加载、显存管理和优雅降级。
    
    通用参数:
    - pitch_shift: 半音调整
    - vpm: 音素周期匹配
    - rms_mix: 响度混合
    """
    
    # 类属性：支持的 f0 方法
    SUPPORTED_F0_METHODS: List[F0Method] = []
    
    # 类属性：是否需要索引文件
    REQUIRE_INDEX: bool = False
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化声音转换器
        
        Args:
            device: 运行设备 ('cpu', 'cuda', 'mps')
        """
        self.device = device or self._get_default_device()
        self._model = None
        self._config = None
        self._is_loaded = False
        self._model_info: Optional[ModelInfo] = None
    
    @abstractmethod
    def load_model(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        index_path: Optional[str] = None,
        **kwargs
    ) -> ModelInfo:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径 (.pth)
            config_path: 配置文件路径 (.json)
            index_path: 索引文件路径 (.index)
            **kwargs: 其他参数
            
        Returns:
            ModelInfo: 模型信息
            
        Raises:
            FileNotFoundError: 模型文件不存在
            ValueError: 模型格式错误
        """
        pass
    
    @abstractmethod
    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        params: Optional[ConversionParams] = None,
        **kwargs
    ) -> ConversionResult:
        """
        执行声音转换
        
        Args:
            audio: 输入音频 (samples,) 或 (channels, samples)
            sample_rate: 输入采样率
            params: 转换参数
            **kwargs: 其他参数覆盖
            
        Returns:
            ConversionResult: 转换结果
        """
        pass
    
    @abstractmethod
    def unload(self):
        """
        卸载模型，释放显存
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Optional[ModelInfo]:
        """获取当前模型信息"""
        pass
    
    def convert_file(
        self,
        input_path: str,
        output_path: str,
        params: Optional[ConversionParams] = None,
        **kwargs
    ) -> bool:
        """
        转换音频文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            params: 转换参数
            **kwargs: 其他参数
            
        Returns:
            bool: 是否成功
        """
        from src.utils.audio_io import AudioLoader, AudioSaver
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 加载音频
        loader = AudioLoader(target_sr=params.sample_rate if params else 40000)
        audio, sr = loader.load(input_path)
        
        # 执行转换
        result = self.convert(audio, sr, params, **kwargs)
        
        # 保存音频
        saver = AudioSaver(normalize=True)
        return saver.save(result.audio, output_path, result.sampling_rate)
    
    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    def _get_default_device(self) -> str:
        """获取默认运行设备"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        验证并规范化音频输入
        
        Args:
            audio: 输入音频
            
        Returns:
            规范化的音频数组 (samples,)
        """
        audio = np.array(audio, dtype=np.float32)
        
        # 处理多通道 -> 单声道
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        elif audio.ndim > 2:
            raise ValueError(f"Invalid audio shape: {audio.shape}")
        
        # 确保是 1D
        if audio.ndim != 1:
            audio = audio.flatten()
        
        return audio
    
    def _validate_params(self, params: Optional[ConversionParams]) -> ConversionParams:
        """验证并规范化参数"""
        if params is None:
            params = ConversionParams()
        
        # 限制 pitch_shift 范围
        params.pitch_shift = max(-24, min(24, params.pitch_shift))
        
        # 限制 vpm 范围
        params.vpm = max(0.0, min(1.0, params.vpm))
        
        # 限制 rms_mix 范围
        params.rms_mix = max(0.0, min(1.0, params.rms_mix))
        
        return params
    
    def _create_result(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> ConversionResult:
        """创建标准结果对象"""
        # 计算基本信息
        duration = len(audio) / sample_rate if audio is not None else 0
        rms = self._calculate_rms(audio)
        
        return ConversionResult(
            audio=audio,
            sampling_rate=sample_rate,
            duration=duration,
            rms_db=rms,
            **kwargs
        )
    
    def _calculate_rms(self, audio: np.ndarray) -> float:
        """计算 RMS 电平(dB)"""
        if audio is None or len(audio) == 0:
            return -float('inf')
        
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return 20 * np.log10(rms)
        return -float('inf')
    
    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "empty"
        return f"{self.__class__.__name__}({self.device}, {status})"
    
    def __enter__(self) -> "BaseVoiceConverter":
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动卸载"""
        self.unload()
        return False


class LazyImportMixin:
    """
    延迟导入混入类
    
    提供延迟导入深度学习依赖的能力，
    避免在模块加载时就检查所有依赖。
    每个子类实例有独立的缓存，避免互相干扰。
    """
    
    # 子类应设置此属性列出需要的模块
    REQUIRED_PACKAGES: List[str] = []
    
    # 缓存已检查的包状态 - 实例变量
    _package_cache: Dict[str, Optional[bool]] = {}
    
    def __init__(self):
        # 每个实例独立的缓存
        if not hasattr(self, '_instance_cache'):
            self._instance_cache: Dict[str, Optional[bool]] = {}
    
    @classmethod
    def _check_dependency(cls, package: str) -> bool:
        """
        检查依赖是否可用（类级别，检查共享缓存）
        
        Args:
            package: 包名
            
        Returns:
            bool: 是否可用
        """
        if package in cls._package_cache:
            return cls._package_cache[package]
        
        try:
            __import__(package)
            cls._package_cache[package] = True
            return True
        except ImportError:
            cls._package_cache[package] = False
            return False
    
    @classmethod
    def _check_dependency_instance(cls, package: str) -> bool:
        """
        检查依赖是否可用（实例级别，每个实例独立缓存）
        
        Args:
            package: 包名
            
        Returns:
            bool: 是否可用
        """
        # 先检查实例缓存
        if hasattr(cls, '_instance_cache') and package in cls._instance_cache:
            return cls._instance_cache[package]
        
        # 检查类缓存
        if package in cls._package_cache:
            result = cls._package_cache[package]
        else:
            try:
                __import__(package)
                cls._package_cache[package] = True
                result = True
            except ImportError:
                cls._package_cache[package] = False
                result = False
        
        # 存入实例缓存
        if hasattr(cls, '_instance_cache'):
            cls._instance_cache[package] = result
        
        return result
    
    @classmethod
    def _lazy_import_module(cls, package: str, submodule: Optional[str] = None):
        """
        延迟导入模块
        
        Args:
            package: 包名
            submodule: 子模块名
            
        Returns:
            导入的模块
            
        Raises:
            ImportError: 模块不可用
        """
        if not cls._check_dependency(package):
            raise ImportError(
                f"Required package '{package}' is not installed.\n"
                f"Please install it with: uv add {package}"
            )
        
        if submodule:
            return __import__(f"{package}.{submodule}", fromlist=[submodule])
        return __import__(package)
    
    @classmethod
    def _check_all_dependencies(cls) -> List[str]:
        """检查所有依赖，返回缺失的包列表"""
        missing = []
        for pkg in cls.REQUIRED_PACKAGES:
            if not cls._check_dependency(pkg):
                missing.append(pkg)
        return missing


class EngineCapability:
    """引擎能力描述"""
    
    # 是否支持 f0 调整
    SUPPORTS_F0: bool = True
    
    # 是否支持音色保护
    SUPPORTS_TIMBRE_PROTECTION: bool = True
    
    # 是否支持扩散模式
    SUPPORTS_DIFFUSION: bool = False
    
    # 是否支持说话人嵌入
    SUPPORTS_SPEAKER_EMBEDDING: bool = False
    
    # 最大支持采样率
    MAX_SAMPLE_RATE: int = 48000
    
    # 推荐采样率
    RECOMMENDED_SAMPLE_RATE: int = 40000
