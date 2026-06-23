"""
Base Effect - 音效处理器基类
定义所有音效处理的通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


@dataclass
class EffectResult:
    """音效处理结果"""
    audio: np.ndarray                    # 处理后的音频
    sample_rate: int = 44100             # 采样率
    parameters_used: Optional[dict] = None  # 使用的参数
    metadata: Optional[dict] = None     # 元数据
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "audio_shape": self.audio.shape if self.audio is not None else None,
            "sample_rate": self.sample_rate,
            "parameters_used": self.parameters_used,
            "metadata": self.metadata,
        }


class BaseEffect(ABC):
    """
    音效处理器基类
    
    所有音效处理器都应继承此类并实现 process 方法。
    支持音频数组的直接处理和文件处理。
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        初始化音效处理器
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
    
    @abstractmethod
    def process(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 44100,
        **kwargs
    ) -> EffectResult:
        """
        处理音频
        
        Args:
            audio: 输入音频数据
            sample_rate: 采样率
            **kwargs: 效果特定参数
            
        Returns:
            EffectResult: 处理结果
        """
        pass
    
    @abstractmethod
    def get_effect_name(self) -> str:
        """获取效果名称"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> dict:
        """获取效果参数"""
        pass
    
    def validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        验证音频输入
        
        Args:
            audio: 输入音频
            
        Returns:
            验证后的音频数组
        """
        audio = np.array(audio, dtype=np.float32)
        
        # 处理单声道
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        elif audio.ndim == 2:
            if audio.shape[0] < audio.shape[1]:
                audio = audio.T
        
        return audio
    
    def apply_with_bypass(
        self,
        audio: np.ndarray,
        sample_rate: int,
        bypass: bool = False,
        **kwargs
    ) -> EffectResult:
        """
        应用效果，支持旁路模式
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            bypass: 是否旁路
            **kwargs: 其他参数
            
        Returns:
            EffectResult: 处理结果
        """
        if bypass:
            return EffectResult(
                audio=self.validate_audio(audio),
                sample_rate=sample_rate,
                parameters_used={"bypass": True},
            )
        
        return self.process(audio, sample_rate, **kwargs)
    
    def _create_result(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> EffectResult:
        """创建标准结果对象"""
        return EffectResult(
            audio=audio,
            sample_rate=sample_rate,
            parameters_used=self.get_parameters(),
            metadata={**kwargs} if kwargs else None,
        )
