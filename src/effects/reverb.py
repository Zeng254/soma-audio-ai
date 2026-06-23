"""
Reverb - 混响效果处理器
实现房间混响、厅堂混响、 Plate 混响等多种类型
"""

from typing import Optional, Tuple
import numpy as np
from scipy import signal

from .base import BaseEffect, EffectResult


class Reverb(BaseEffect):
    """
    混响效果处理器
    
    基于 Schroeder/Moorer 模型的数字混响实现。
    
    混响参数:
    - room_size: 房间大小 (0-1)
    - damping: 阻尼 (0-1)
    - wet_level: 混响信号电平 (0-1)
    - dry_level: 干信号电平 (0-1)
    - width: 立体声宽度 (0-1)
    
    混响类型:
    - room: 房间混响
    - hall: 厅堂混响
    - plate: Plate 混响
    - cathedral: 大教堂混响
    """
    
    # 梳状滤波器延迟线长度 (采样点数 @ 44100Hz)
    COMB_DELAYS = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
    
    # 全通滤波器延迟线长度
    ALLPASS_DELAYS = [225, 556, 441, 341]
    
    def __init__(
        self,
        sample_rate: int = 44100,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3,
        dry_level: float = 0.7,
        width: float = 1.0,
        reverb_type: str = "room",
    ):
        """
        初始化混响处理器
        
        Args:
            sample_rate: 采样率
            room_size: 房间大小 (0-1)
            damping: 高频阻尼 (0-1)
            wet_level: 混响信号电平
            dry_level: 干信号电平
            width: 立体声宽度
            reverb_type: 混响类型
        """
        super().__init__(sample_rate)
        self.room_size = max(0.0, min(1.0, room_size))
        self.damping = max(0.0, min(1.0, damping))
        self.wet_level = max(0.0, min(1.0, wet_level))
        self.dry_level = max(0.0, min(1.0, dry_level))
        self.width = max(0.0, min(1.0, width))
        self.reverb_type = reverb_type
        
        # 初始化滤波器状态
        self._comb_buffers = None
        self._allpass_buffers = None
        self._init_buffers()
    
    def _init_buffers(self):
        """初始化延迟线缓冲区"""
        # 梳状滤波器缓冲区
        self._comb_buffers = [
            np.zeros(delay) for delay in self.COMB_DELAYS
        ]
        self._comb_indices = [0] * len(self.COMB_DELAYS)
        
        # 全通滤波器缓冲区
        self._allpass_buffers = [
            np.zeros(delay) for delay in self.ALLPASS_DELAYS
        ]
        self._allpass_indices = [0] * len(self.ALLPASS_DELAYS)
    
    def get_effect_name(self) -> str:
        return f"Reverb-{self.reverb_type}"
    
    def get_parameters(self) -> dict:
        return {
            "room_size": self.room_size,
            "damping": self.damping,
            "wet_level": self.wet_level,
            "dry_level": self.dry_level,
            "width": self.width,
            "reverb_type": self.reverb_type,
        }
    
    def _adjust_for_type(self):
        """根据混响类型调整参数"""
        presets = {
            "room": {
                "room_size": 0.4,
                "damping": 0.6,
                "comb_feedback": 0.7,
            },
            "hall": {
                "room_size": 0.8,
                "damping": 0.3,
                "comb_feedback": 0.85,
            },
            "plate": {
                "room_size": 0.6,
                "damping": 0.8,
                "comb_feedback": 0.75,
            },
            "cathedral": {
                "room_size": 0.95,
                "damping": 0.2,
                "comb_feedback": 0.92,
            },
        }
        
        if self.reverb_type in presets:
            preset = presets[self.reverb_type]
            self.room_size = preset["room_size"]
            self.damping = preset["damping"]
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        **kwargs
    ) -> EffectResult:
        """
        应用混响效果
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            **kwargs: 可选参数覆盖
            
        Returns:
            EffectResult: 处理结果
        """
        audio = self.validate_audio(audio)
        self.sample_rate = sample_rate
        
        # 更新参数
        for key in ["room_size", "damping", "wet_level", "dry_level", "width", "reverb_type"]:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        
        # 根据类型调整
        self._adjust_for_type()
        
        # 重新初始化缓冲区（如果采样率改变）
        if self._comb_buffers is None or len(self._comb_buffers[0]) != int(self.COMB_DELAYS[0] * sample_rate / 44100):
            self._init_buffers()
        
        # 缩放延迟以适应采样率
        scale = sample_rate / 44100
        
        # 处理每个通道
        is_stereo = audio.shape[0] >= 2
        
        if is_stereo:
            # 立体声处理
            left = self._process_channel(audio[0], scale)
            right = self._process_channel(audio[1], scale)
            
            # 应用立体声宽度
            if self.width < 1.0:
                mid = (left + right) / 2
                left = mid + self.width * (left - mid)
                right = mid + self.width * (right - mid)
            
            result = np.vstack([left, right])
        else:
            # 单声道处理
            result = self._process_channel(audio[0], scale)
            result = result[np.newaxis, :]
        
        return self._create_result(result, sample_rate)
    
    def _process_channel(self, audio: np.ndarray, scale: float) -> np.ndarray:
        """处理单个声道"""
        output = np.zeros_like(audio)
        
        # 计算梳状滤波器反馈
        feedback = 0.7 + (self.room_size * 0.28)
        
        # 处理每个采样点
        for i, sample in enumerate(audio):
            reverb_sample = 0.0
            
            # 并行梳状滤波器
            for j, delay in enumerate(self.COMB_DELAYS):
                actual_delay = int(delay * scale)
                if actual_delay > 0 and actual_delay <= len(self._comb_buffers[j]):
                    idx = (self._comb_indices[j] - actual_delay) % len(self._comb_buffers[j])
                    reverb_sample += self._comb_buffers[j][idx]
            
            # 归一化
            reverb_sample /= len(self.COMB_DELAYS)
            
            # 写入缓冲区
            for j in range(len(self._comb_buffers)):
                self._comb_buffers[j][self._comb_indices[j]] = (
                    sample + reverb_sample * feedback * (1 - self.damping * 0.5)
                )
                self._comb_indices[j] = (self._comb_indices[j] + 1) % len(self._comb_buffers[j])
            
            # 级联全通滤波器
            temp = reverb_sample
            for j, delay in enumerate(self.ALLPASS_DELAYS):
                actual_delay = int(delay * scale)
                if actual_delay > 0 and actual_delay <= len(self._allpass_buffers[j]):
                    idx = (self._allpass_indices[j] - actual_delay) % len(self._allpass_buffers[j])
                    temp2 = self._allpass_buffers[j][idx]
                    self._allpass_buffers[j][self._allpass_indices[j]] = temp
                    self._allpass_indices[j] = (self._allpass_indices[j] + 1) % len(self._allpass_buffers[j])
                    temp = -0.5 * temp + temp2 + 0.5 * self._allpass_buffers[j][idx]
            
            output[i] = sample * self.dry_level + temp * self.wet_level
        
        return output
    
    def _freeverb_process(self, audio: np.ndarray) -> np.ndarray:
        """
        Freeverb 风格混响处理
        
        这是 Schroeder 模型的优化实现
        """
        # 梳状滤波器数量
        num_combs = 8
        num_allpasses = 4
        
        # 梳状滤波器延迟时间（采样）
        comb_tunings = [1116, 1188, 1277, 1356, 1422, 1491, 1617, 1557]
        allpass_tunings = [556, 441, 341, 225]
        
        # 调整到当前采样率
        scale = self.sample_rate / 44100
        comb_delays = [int(d * scale) for d in comb_tunings]
        allpass_delays = [int(d * scale) for d in allpass_tunings]
        
        # 初始化缓冲区
        comb_buffers = [np.zeros(d) for d in comb_delays]
        allpass_buffers = [np.zeros(d) for d in allpass_delays]
        comb_indices = [0] * num_combs
        allpass_indices = [0] * num_allpasses
        
        # 计算参数
        feedback = 0.015 + self.room_size * 0.25
        damping = self.damping * 0.4
        
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # 获取梳状滤波器输出和
            comb_sum = 0
            for c in range(num_combs):
                buf = comb_buffers[c]
                idx = comb_indices[c]
                output_val = buf[idx]
                
                # 低通滤波
                filtered = output_val * (1 - damping) + comb_sum * damping
                comb_sum += filtered
                
                # 写入新样本
                buf[idx] = audio[i] + filtered * feedback
                comb_indices[c] = (idx + 1) % len(buf)
            
            comb_sum /= num_combs
            
            # 全通滤波器级联
            for a in range(num_allpasses):
                buf = allpass_buffers[a]
                idx = allpass_indices[a]
                temp = buf[idx]
                buf[idx] = comb_sum
                allpass_indices[a] = (idx + 1) % len(buf)
                comb_sum = -0.5 * comb_sum + temp + 0.5 * buf[idx]
            
            output[i] = audio[i] * self.dry_level + comb_sum * self.wet_level
        
        return output
