"""
Base Separator - 音频分离器基类
定义所有分离器的通用接口和抽象方法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import numpy as np


@dataclass
class SeparationResult:
    """音频分离结果"""
    vocals: Optional[np.ndarray] = None      # 人声
    accompaniment: Optional[np.ndarray] = None  # 伴奏
    drums: Optional[np.ndarray] = None      # 鼓点
    bass: Optional[np.ndarray] = None       # 贝斯
    other: Optional[np.ndarray] = None       # 其他
    sample_rate: int = 44100                 # 采样率
    
    def get_track(self, name: str) -> Optional[np.ndarray]:
        """根据名称获取分离后的音轨"""
        track_map = {
            "vocals": self.vocals,
            "accompaniment": self.accompaniment,
            "drums": self.drums,
            "bass": self.bass,
            "other": self.other,
        }
        return track_map.get(name)
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "vocals": self.vocals is not None,
            "accompaniment": self.accompaniment is not None,
            "drums": self.drums is not None,
            "bass": self.bass is not None,
            "other": self.other is not None,
            "sample_rate": self.sample_rate,
        }


class BaseSeparator(ABC):
    """
    音频分离器基类
    
    定义音频分离任务的通用接口，支持：
    - 人声/伴奏分离
    - 多轨道分离（鼓、贝斯、其他）
    - 去混响
    - 降噪
    """
    
    def __init__(self, sample_rate: int = 44100, device: Optional[str] = None):
        """
        初始化分离器
        
        Args:
            sample_rate: 目标采样率
            device: 运行设备 ('cpu', 'cuda', 'mps')
        """
        self.sample_rate = sample_rate
        self.device = device or self._get_default_device()
    
    @abstractmethod
    def separate(self, audio_path: str, **kwargs) -> SeparationResult:
        """
        执行音频分离
        
        Args:
            audio_path: 输入音频文件路径
            **kwargs: 其他参数
            
        Returns:
            SeparationResult: 分离结果
        """
        pass
    
    @abstractmethod
    def separate_array(self, audio: np.ndarray, sample_rate: int = 44100, **kwargs) -> SeparationResult:
        """
        对音频数组进行分离
        
        Args:
            audio: 音频数据 (samples, channels) 或 (channels, samples)
            sample_rate: 采样率
            **kwargs: 其他参数
            
        Returns:
            SeparationResult: 分离结果
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """获取模型名称"""
        pass
    
    @abstractmethod
    def get_available_tracks(self) -> List[str]:
        """获取可分离的音轨列表"""
        pass
    
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
    
    def validate_audio_input(self, audio: np.ndarray) -> np.ndarray:
        """
        验证并规范化音频输入
        
        Args:
            audio: 输入音频
            
        Returns:
            规范化的音频数组 (channels, samples)
        """
        # 确保是 numpy 数组
        audio = np.array(audio, dtype=np.float32)
        
        # 处理单声道
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        elif audio.ndim == 2:
            # 确保格式为 (channels, samples)
            if audio.shape[0] < audio.shape[1]:
                audio = audio.T
        else:
            raise ValueError(f"Invalid audio shape: {audio.shape}")
        
        return audio
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        归一化音频电平
        
        Args:
            audio: 输入音频
            target_db: 目标分贝值
            
        Returns:
            归一化后的音频
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_linear = 10 ** (target_db / 20)
            audio = audio * (target_linear / max_val)
        return audio
