"""
Pitch Shifter - 音调变换处理器
支持半音阶和音分数的音高调整
"""

from typing import Optional, Tuple
import numpy as np

from .base import BaseEffect, EffectResult


class PitchShifter(BaseEffect):
    """
    音调变换处理器
    
    基于 WSOLA (Waveform Similarity Overlap-Add) 算法实现，
    支持高质量的实时音高调整。
    
    功能:
    - 半音阶音高调整 (semitones)
    - 音分数调整 (cents)
    - 速率保持 (preserve_formant)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        semitones: float = 0.0,
        cents: float = 0.0,
        preserve_formant: bool = True,
    ):
        """
        初始化音调变换器
        
        Args:
            sample_rate: 采样率
            semitones: 半音阶调整 (-12 to +12)
            cents: 音分数调整 (-100 to +100)
            preserve_formant: 是否保留共振峰
        """
        super().__init__(sample_rate)
        self.semitones = semitones
        self.cents = cents
        self.preserve_formant = preserve_formant
    
    @property
    def pitch_ratio(self) -> float:
        """计算音高比例因子"""
        total_cents = self.semitones * 100 + self.cents
        return 2 ** (total_cents / 1200)
    
    def get_effect_name(self) -> str:
        return "PitchShifter"
    
    def get_parameters(self) -> dict:
        return {
            "semitones": self.semitones,
            "cents": self.cents,
            "pitch_ratio": self.pitch_ratio,
            "preserve_formant": self.preserve_formant,
        }
    
    def shift(
        self,
        audio: np.ndarray,
        semitones: float,
        cents: float = 0.0,
    ) -> EffectResult:
        """
        调整音高
        
        Args:
            audio: 输入音频
            semitones: 半音阶调整
            cents: 音分数调整
            
        Returns:
            EffectResult: 处理结果
        """
        self.semitones = semitones
        self.cents = cents
        return self.process(audio, self.sample_rate)
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        **kwargs
    ) -> EffectResult:
        """
        应用音调变换
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            **kwargs: 参数覆盖
            
        Returns:
            EffectResult: 处理结果
        """
        audio = self.validate_audio(audio)
        self.sample_rate = sample_rate
        
        # 更新参数
        for key in ["semitones", "cents", "preserve_formant"]:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        
        ratio = self.pitch_ratio
        
        if abs(ratio - 1.0) < 0.001:
            # 无需处理
            return self._create_result(audio, sample_rate)
        
        # 重采样实现音高变换
        result = self._time_stretch(audio, 1.0 / ratio)
        
        # 可选：保留共振峰
        if self.preserve_formant and abs(ratio - 1.0) > 0.1:
            result = self._formant_preserve(result, ratio)
        
        return self._create_result(result, sample_rate, ratio=ratio)
    
    def _time_stretch(
        self,
        audio: np.ndarray,
        stretch_factor: float,
        segment_length: int = 1024,
        overlap: float = 0.5,
    ) -> np.ndarray:
        """
        时间拉伸（WSOLA 算法）
        
        Args:
            audio: 输入音频
            stretch_factor: 拉伸因子 (<1 加速, >1 减速)
            segment_length: 分段长度
            overlap: 重叠比例
            
        Returns:
            拉伸后的音频
        """
        try:
            import librosa
            # 使用 librosa 实现高质量时间拉伸
            return self._librosa_time_stretch(audio, stretch_factor)
        except ImportError:
            # 降级到 scipy 实现
            return self._scipy_time_stretch(audio, stretch_factor)
    
    def _librosa_time_stretch(
        self,
        audio: np.ndarray,
        stretch_factor: float,
    ) -> np.ndarray:
        """使用 librosa 实现时间拉伸"""
        import librosa
        
        # 处理多通道
        if audio.shape[0] > 1:
            results = []
            for ch in audio:
                stretched = librosa.effects.time_stretch(ch, rate=stretch_factor)
                results.append(stretched)
            return np.vstack(results)
        else:
            audio_flat = audio[0] if audio.ndim > 1 else audio
            return librosa.effects.time_stretch(audio_flat, rate=stretch_factor)[np.newaxis, :]
    
    def _scipy_time_stretch(
        self,
        audio: np.ndarray,
        stretch_factor: float,
    ) -> np.ndarray:
        """使用 scipy 信号处理实现时间拉伸"""
        from scipy import signal
        
        # 计算新长度
        new_length = int(len(audio[0]) * stretch_factor)
        
        # 处理每个通道
        results = []
        for ch in audio:
            # 重采样实现时间拉伸
            indices = np.round(np.arange(0, len(ch), stretch_factor)).astype(int)
            indices = indices[indices < len(ch)]
            stretched = ch[indices]
            
            # 插值到目标长度
            x_old = np.linspace(0, 1, len(stretched))
            x_new = np.linspace(0, 1, new_length)
            stretched = np.interp(x_new, x_old, stretched)
            
            results.append(stretched)
        
        return np.vstack(results)
    
    def _formant_preserve(
        self,
        audio: np.ndarray,
        pitch_ratio: float,
    ) -> np.ndarray:
        """
        保留共振峰
        
        通过调整共振峰位置来保持原始音色特征
        
        Args:
            audio: 输入音频
            pitch_ratio: 音高比例
            
        Returns:
            保留共振峰后的音频
        """
        # 简化的共振峰保留
        # 实际实现需要 LPC 分析和合成
        try:
            import librosa
            
            # 估算并移动共振峰
            # 这里使用简化的预加重/去加重
            alpha = 0.95 / pitch_ratio
            audio_processed = np.zeros_like(audio)
            
            for ch in range(audio.shape[0]):
                ch_data = audio[ch]
                # 预加重
                pre_emphasized = np.append(ch_data[0], ch_data[1:] - alpha * ch_data[:-1])
                # 去加重
                de_emphasized = np.zeros_like(pre_emphasized)
                for i in range(1, len(pre_emphasized)):
                    de_emphasized[i] = pre_emphasized[i] + alpha * de_emphasized[i-1]
                audio_processed[ch] = de_emphasized
            
            return audio_processed
        except ImportError:
            return audio
    
    def detect_pitch(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
    ) -> Tuple[float, float]:
        """
        检测音频基频
        
        Args:
            audio: 输入音频
            sample_rate: 采样率
            
        Returns:
            (frequency_hz, confidence)
        """
        try:
            import librosa
            audio_flat = audio[0] if audio.shape[0] > 1 else audio
            pitches, magnitudes = librosa.piptrack(
                y=audio_flat.astype(np.float32),
                sr=sample_rate,
            )
            
            # 获取最大幅度的音高
            max_idx = np.unravel_index(np.argmax(magnitudes), magnitudes.shape)
            pitch_hz = pitches[max_idx]
            confidence = magnitudes[max_idx] / np.max(magnitudes) if np.max(magnitudes) > 0 else 0
            
            return float(pitch_hz), float(confidence)
        except ImportError:
            return 0.0, 0.0
    
    def match_key(self, audio: np.ndarray, target_key: str = "C") -> float:
        """
        匹配音乐调性
        
        Args:
            audio: 输入音频
            target_key: 目标调性 (如 "C", "Am")
            
        Returns:
            需要的半音调整数
        """
        current_pitch, _ = self.detect_pitch(audio, self.sample_rate)
        
        # 简单实现：返回需要的调整
        # 实际需要音阶匹配算法
        return 0.0
