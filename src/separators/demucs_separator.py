"""
Demucs Separator - 基于 Demucs 的音频分离器
支持人声/伴奏/鼓/贝斯/其他的四轨分离
"""

from typing import Optional
from pathlib import Path
import numpy as np

from .base import BaseSeparator, SeparationResult


class DemucsSeparator(BaseSeparator):
    """
    Demucs 音频分离器
    
    基于 Facebook Research 的 Demucs 深度学习模型，
    支持高质量的音频源分离。
    
    支持的模型:
    - hdemucs.mmi: 通用模型，支持 4 轨道分离
    -htdemucs: 通用模型，支持 4 轨道分离
    - htdemucs_ft: 微调模型，更高质量但更慢
    - htdemucs_mmi: MMI 版本
    
    支持的音轨:
    - vocals: 人声
    - drums: 鼓点
    - bass: 贝斯
    - other: 其他乐器
    """
    
    def __init__(
        self,
        model_name: str = "htdemucs",
        sample_rate: int = 44100,
        device: Optional[str] = None,
        progress: bool = True,
    ):
        """
        初始化 Demucs 分离器
        
        Args:
            model_name: 模型名称
            sample_rate: 采样率
            device: 运行设备
            progress: 是否显示进度
        """
        super().__init__(sample_rate, device)
        self.model_name = model_name
        self.progress = progress
        self._model = None
        self._demucs = None
    
    def _load_model(self):
        """延迟加载 Demucs 模型"""
        if self._model is None:
            try:
                from demucs import pretrained
                from demucs.pretrained import get_model
                self._model = get_model(self.model_name)
                if self.device != "cpu":
                    self._model.to(self.device)
                self._model.eval()
            except ImportError:
                raise ImportError(
                    "Demucs not installed. Install with: uv add demucs"
                )
    
    def get_model_name(self) -> str:
        return f"Demucs-{self.model_name}"
    
    def get_available_tracks(self) -> list:
        return ["vocals", "drums", "bass", "other"]
    
    def separate(self, audio_path: str, **kwargs) -> SeparationResult:
        """
        从文件路径分离音频
        
        Args:
            audio_path: 输入音频文件路径
            **kwargs: 其他参数 (如 output_dir)
            
        Returns:
            SeparationResult: 分离结果
        """
        from utils.audio_io import AudioLoader
        
        loader = AudioLoader()
        audio, sr = loader.load(audio_path)
        
        return self.separate_array(audio, sr, **kwargs)
    
    def separate_array(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 44100,
        **kwargs
    ) -> SeparationResult:
        """
        对音频数组进行分离
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            **kwargs: 其他参数
            
        Returns:
            SeparationResult: 分离结果
        """
        self._load_model()
        
        # 验证并规范化输入
        audio = self.validate_audio_input(audio)
        
        # 重采样到模型需要的采样率
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
        
        # 转换为模型输入格式
        audio_tensor = self._prepare_tensor(audio)
        
        # 执行分离
        with np.errstate(divide='ignore', invalid='ignore'):
            separated = self._model(audio_tensor)
        
        # 提取各音轨
        result = SeparationResult(sample_rate=self.sample_rate)
        
        tracks = ["vocals", "drums", "bass", "other"]
        track_arrays = [result.vocals, result.drums, result.bass, result.other]
        
        for i, (track_name, track_array) in enumerate(zip(tracks, track_arrays)):
            if i < separated.shape[1]:
                track_data = separated[0, i].cpu().numpy()
                if track_name == "vocals":
                    result.vocals = track_data
                elif track_name == "drums":
                    result.drums = track_data
                elif track_name == "bass":
                    result.bass = track_data
                elif track_name == "other":
                    result.other = track_data
        
        return result
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """重采样音频"""
        try:
            import librosa
            if audio.shape[0] == 2:
                audio = librosa.to_mono(audio)
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            from scipy import signal
            num_samples = int(len(audio[0]) * target_sr / orig_sr)
            return signal.resample(audio, num_samples, axis=-1)
    
    def _prepare_tensor(self, audio: np.ndarray) -> "torch.Tensor":
        """准备模型输入张量"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Install with: uv add torch torchaudio")
        
        # 转换为张量 (batch, channels, samples)
        tensor = torch.from_numpy(audio).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        
        # 移动到设备
        tensor = tensor.to(self.device)
        
        return tensor
