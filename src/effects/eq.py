"""
Equalizer - 多频段均衡器
支持自定义频段增益调整
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy import signal

from .base import BaseEffect, EffectResult


class Equalizer(BaseEffect):
    """
    多频段均衡器
    
    支持任意数量的频段设置，每个频段包含：
    - 中心频率 (Hz)
    - 增益 (dB)
    - 带宽 (Q值)
    
    支持的滤波器类型:
    - peak: 峰值/陷波滤波器
    - lowshelf: 低架滤波器
    - highshelf: 高架滤波器
    - lowpass: 低通滤波器
    - highpass: 高通滤波器
    - bandpass: 带通滤波器
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        bands: Optional[List[dict]] = None,
    ):
        """
        初始化均衡器
        
        Args:
            sample_rate: 采样率
            bands: 预设频段列表
                 示例: [{"freq": 60, "gain": 3, "q": 1.4, "type": "peak"}, ...]
        """
        super().__init__(sample_rate)
        self.bands = bands or self._default_bands()
    
    def _default_bands(self) -> List[dict]:
        """默认 10 段均衡器预设"""
        return [
            {"freq": 32, "gain": 0, "q": 1.4, "type": "lowshelf"},
            {"freq": 64, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 125, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 250, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 500, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 1000, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 2000, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 4000, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 8000, "gain": 0, "q": 1.4, "type": "peak"},
            {"freq": 16000, "gain": 0, "q": 1.4, "type": "highshelf"},
        ]
    
    def get_effect_name(self) -> str:
        return "Equalizer"
    
    def get_parameters(self) -> dict:
        return {"bands": self.bands, "sample_rate": self.sample_rate}
    
    def set_band(
        self,
        index: int,
        freq: float,
        gain: float,
        q: float = 1.4,
        filter_type: str = "peak"
    ):
        """
        设置单个频段
        
        Args:
            index: 频段索引
            freq: 中心频率
            gain: 增益(dB)
            q: 带宽
            filter_type: 滤波器类型
        """
        if 0 <= index < len(self.bands):
            self.bands[index] = {
                "freq": freq,
                "gain": gain,
                "q": q,
                "type": filter_type,
            }
    
    def set_preset(self, name: str):
        """
        应用预设
        
        Args:
            name: 预设名称
        """
        presets = {
            "flat": self._flat_preset(),
            "bass_boost": self._bass_boost_preset(),
            "treble_boost": self._treble_boost_preset(),
            "vocal_boost": self._vocal_boost_preset(),
            "pop": self._pop_preset(),
            "rock": self._rock_preset(),
            "jazz": self._jazz_preset(),
            "classical": self._classical_preset(),
        }
        
        if name in presets:
            self.bands = presets[name]
        else:
            raise ValueError(f"Unknown preset: {name}")
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        **kwargs
    ) -> EffectResult:
        """
        应用均衡器
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            **kwargs: 可选覆盖参数
            
        Returns:
            EffectResult: 处理结果
        """
        audio = self.validate_audio(audio)
        self.sample_rate = sample_rate
        
        # 合并传入的频段设置
        bands = kwargs.get("bands", self.bands)
        
        # 对每个通道应用均衡器
        result = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            filtered = audio[ch].copy()
            for band in bands:
                filtered = self._apply_band(filtered, band)
            result[ch] = filtered
        
        return self._create_result(result, sample_rate, bands=bands)
    
    def _apply_band(self, audio: np.ndarray, band: dict) -> np.ndarray:
        """应用单个频段"""
        freq = band["freq"]
        gain = band["gain"]
        q = band["q"]
        filter_type = band["type"]
        
        # 计算 b, a 系数
        b, a = self._design_filter(freq, gain, q, filter_type)
        
        # 应用滤波器
        return signal.lfilter(b, a, audio)
    
    def _design_filter(
        self,
        freq: float,
        gain: float,
        q: float,
        filter_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        设计 IIR 滤波器系数
        
        Args:
            freq: 中心频率
            gain: 增益(dB)
            q: 品质因数
            filter_type: 滤波器类型
            
        Returns:
            (b, a): 滤波器系数
        """
        A = 10 ** (gain / 40)
        w0 = 2 * np.pi * freq / self.sample_rate
        
        if filter_type == "peak":
            alpha = np.sin(w0) / (2 * q)
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
            
        elif filter_type == "lowshelf":
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/q - 1) + 2)
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            
        elif filter_type == "highshelf":
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/q - 1) + 2)
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            
        else:
            # 默认使用 peak 滤波器
            return self._design_filter(freq, gain, q, "peak")
        
        # 归一化系数
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])
        
        return b, a
    
    # 预设方法
    def _flat_preset(self) -> List[dict]:
        bands = self._default_bands()
        for band in bands:
            band["gain"] = 0
        return bands
    
    def _bass_boost_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = 6
        bands[1]["gain"] = 5
        bands[2]["gain"] = 4
        bands[3]["gain"] = 2
        return bands
    
    def _treble_boost_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[-1]["gain"] = 6
        bands[-2]["gain"] = 5
        bands[-3]["gain"] = 4
        bands[-4]["gain"] = 2
        return bands
    
    def _vocal_boost_preset(self) -> List[dict]:
        bands = self._default_bands()
        for i, band in enumerate(bands):
            if 3 <= i <= 6:  # 250Hz - 1kHz
                band["gain"] = 3
        return bands
    
    def _pop_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = -1
        bands[1]["gain"] = 4
        bands[2]["gain"] = 5
        bands[3]["gain"] = 3
        bands[4]["gain"] = -1
        bands[5]["gain"] = -1
        bands[6]["gain"] = 2
        bands[7]["gain"] = 4
        bands[8]["gain"] = 5
        bands[9]["gain"] = 3
        return bands
    
    def _rock_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = 5
        bands[1]["gain"] = 4
        bands[2]["gain"] = 2
        bands[3]["gain"] = -1
        bands[4]["gain"] = -2
        bands[5]["gain"] = 2
        bands[6]["gain"] = 3
        bands[7]["gain"] = 4
        bands[8]["gain"] = 4
        bands[9]["gain"] = 4
        return bands
    
    def _jazz_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = 3
        bands[1]["gain"] = 2
        bands[2]["gain"] = 1
        bands[3]["gain"] = 2
        bands[4]["gain"] = -2
        bands[5]["gain"] = -2
        bands[6]["gain"] = 0
        bands[7]["gain"] = 2
        bands[8]["gain"] = 3
        bands[9]["gain"] = 4
        return bands
    
    def _classical_preset(self) -> List[dict]:
        bands = self._default_bands()
        bands[0]["gain"] = 5
        bands[1]["gain"] = 4
        bands[2]["gain"] = 3
        bands[3]["gain"] = 2
        bands[4]["gain"] = -1
        bands[5]["gain"] = -1
        bands[6]["gain"] = -1
        bands[7]["gain"] = 0
        bands[8]["gain"] = 2
        bands[9]["gain"] = 3
        return bands
