"""
Demucs Separator - 基于 Demucs 的音频分离器

支持人声/伴奏/鼓/贝斯/其他的四轨分离。
"""

from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from src.separators.base import BaseSeparator, SeparationResult
from src.utils.audio_io import AudioLoader


class DemucsSeparator(BaseSeparator):
    """
    Demucs 音频分离器

    基于 Facebook Research 的 Demucs 深度学习模型，
    支持高质量的音频源分离。

    支持的模型:
    - hdemucs.mmi: 通用模型，支持 4 轨道分离
    - htdemucs: 通用模型，支持 4 轨道分离
    - htdemucs_ft: 微调模型，更高质量但更慢
    - htdemucs_mmi: MMI 版本

    支持的音轨:
    - vocals: 人声
    - drums: 鼓点
    - bass: 贝斯
    - other: 其他乐器
    """

    # Demucs 模型的标准音轨顺序
    DEFAULT_TRACKS = ["vocals", "drums", "bass", "other"]

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
                # Demucs 4.0+ API: get_model 返回一个包含 model 的对象
                model_bundle = get_model(self.model_name)
                # 从 bundle 中获取实际的模型
                self._model = model_bundle
                self._demucs = model_bundle.get_model()
                if self.device != "cpu":
                    self._demucs.to(self.device)
                self._demucs.eval()
            except ImportError:
                raise ImportError(
                    "Demucs 未安装。请运行: uv add demucs"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Demucs model: {e}")

    def get_model_name(self) -> str:
        return f"Demucs-{self.model_name}"

    def get_available_tracks(self) -> list:
        """获取可用的音轨列表"""
        if self._demucs is not None:
            return list(self._demucs.sources)
        return list(self.DEFAULT_TRACKS)

    def separate(self, audio_path: str, **kwargs) -> SeparationResult:
        """
        从文件路径分离音频

        Args:
            audio_path: 输入音频文件路径
            **kwargs: 其他参数 (如 output_dir)

        Returns:
            SeparationResult: 分离结果
        """
        loader = AudioLoader(channel_first=True)
        audio, sr = loader.load(audio_path, force_channel_first=True)

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
            audio: 音频数据，格式为 (channels, samples) 或 (samples,)
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

        # 执行分离 - Demucs 4.0+ API
        import torch
        with torch.no_grad():
            sources = self._demucs(audio_tensor)

        # 提取各音轨到结果对象
        result = self._extract_tracks(separated)

        return result

    def _extract_tracks(self, sources: "torch.Tensor") -> SeparationResult:
        """
        从模型输出提取各音轨

        Args:
            sources: 模型输出张量，形状为 (batch, tracks, samples)
                     tracks 顺序由 Demucs 定义

        Returns:
            SeparationResult: 包含各音轨的结果
        """
        result = SeparationResult(sample_rate=self.sample_rate)

        # 获取 Demucs 定义的音轨顺序
        tracks = self._demucs.sources

        # 确保输出在 CPU 上并转为 numpy
        sources_np = sources[0].cpu().numpy()  # (tracks, samples)

        # 提取各音轨
        for i, track_name in enumerate(tracks):
            if i < len(sources_np):
                track_data = sources_np[i]

                # 直接赋值给 result 对象
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
            if audio.ndim == 2:
                # 转换为 (samples, channels) 进行 librosa 重采样
                audio_t = audio.T
                result = np.zeros((int(audio_t.shape[0] * target_sr / orig_sr), audio_t.shape[1]))
                for ch in range(audio_t.shape[1]):
                    result[:, ch] = librosa.resample(
                        audio_t[:, ch],
                        orig_sr=orig_sr,
                        target_sr=target_sr
                    )
                return result.T  # 转回 (channels, samples)
            else:
                return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            from scipy import signal
            if audio.ndim == 2:
                num_samples = int(audio.shape[1] * target_sr / orig_sr)
                result = np.zeros((audio.shape[0], num_samples))
                for ch in range(audio.shape[0]):
                    result[ch] = signal.resample(audio[ch], num_samples)
                return result
            else:
                num_samples = int(len(audio) * target_sr / orig_sr)
                return signal.resample(audio, num_samples)

    def _prepare_tensor(self, audio: np.ndarray) -> "torch.Tensor":
        """准备模型输入张量"""
        try:
            import torch
        except ImportError:
            raise ImportError("需要 PyTorch。请运行: uv add torch torchaudio")

        # 转换为张量 (batch, channels, samples)
        tensor = torch.from_numpy(audio).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        elif tensor.dim() == 2:
            if audio.shape[0] > audio.shape[1]:
                # (channels, samples) -> (1, channels, samples)
                tensor = tensor.unsqueeze(0)
            else:
                # (samples, channels) -> (1, channels, samples)
                tensor = tensor.T.unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(0)

        # 移动到设备
        tensor = tensor.to(self.device)

        return tensor
