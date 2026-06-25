"""
Audio I/O - 音频读写工具

提供统一的音频文件读写接口。

音频格式约定:
- 内部存储格式: (channels, samples) - 2D 数组
- 文件读写格式: (samples, channels) - soundfile 标准格式
- 库玛索引: (samples,) - 单声道
"""

from pathlib import Path
from typing import Tuple, Optional, Union, List
import numpy as np


class AudioLoader:
    """
    音频加载器

    支持多种音频格式的加载和预处理。
    """

    SUPPORTED_FORMATS = {
        "wav", "mp3", "flac", "ogg", "aac",
        "m4a", "wma", "aiff", "amr", "opus"
    }

    # 常见采样率，用于判断是否为通道数
    COMMON_SAMPLE_RATES = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000}

    def __init__(
        self,
        target_sr: Optional[int] = None,
        mono: bool = False,
        channel_first: Optional[bool] = None
    ):
        """
        初始化加载器

        Args:
            target_sr: 目标采样率，None 表示保持原采样率
            mono: 是否转换为单声道
            channel_first: 明确指定输入/输出格式
                          True: 返回 (channels, samples)
                          False: 返回 (samples, channels)
                          None: 自动检测（默认）
        """
        self.target_sr = target_sr
        self.mono = mono
        self.channel_first = channel_first

    def load(
        self,
        file_path: str,
        force_channel_first: Optional[bool] = None
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频文件

        Args:
            file_path: 音频文件路径
            force_channel_first: 强制指定输出格式

        Returns:
            (audio_data, sample_rate)
            - channel_first=True: (channels, samples)
            - channel_first=False: (samples, channels)
        """
        path = Path(file_path)
        suffix = path.suffix[1:].lower()

        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}")

        # 确定输出格式
        output_channel_first = force_channel_first if force_channel_first is not None else self.channel_first
        if output_channel_first is None:
            output_channel_first = True  # 内部默认使用 channel_first

        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype='float32')
            # soundfile 返回 (samples, channels)

        except ImportError:
            # 降级到 pydub
            from pydub import AudioSegment
            audio_segment = AudioSegment.from_file(str(path))
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            sr = audio_segment.frame_rate

            # pydub 返回 (samples,)，需要 reshape
            if audio_segment.channels == 2:
                audio = audio.reshape((-1, 2))
            else:
                audio = audio.reshape(-1, 1)

        # 检测并转换通道格式
        audio = self._ensure_channel_first(audio, sr)

        # 转换单声道
        if self.mono and audio.shape[0] > 1:
            audio = np.mean(audio, axis=0, keepdims=True)

        # 重采样
        if self.target_sr and self.target_sr != sr:
            audio = self._resample(audio, sr, self.target_sr)
            sr = self.target_sr

        # 根据 output_channel_first 决定是否转置
        if not output_channel_first:
            audio = audio.T

        return audio, sr

    def _ensure_channel_first(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        确保音频格式为 (channels, samples)

        通过多种方式检测当前格式：
        1. 如果有显式的 channel_first 设置，使用该设置
        2. 如果是 1D 数组，直接添加维度
        3. 通过维度大小推断格式

        Args:
            audio: 输入音频数据
            sample_rate: 采样率

        Returns:
            (channels, samples) 格式的音频
        """
        # 如果有显式设置
        if self.channel_first is not None:
            if self.channel_first and audio.ndim == 2 and audio.shape[0] > audio.shape[-1]:
                return audio.T
            elif not self.channel_first and audio.ndim == 2 and audio.shape[0] < audio.shape[-1]:
                return audio.T
            return audio

        # 1D 数组（单声道）
        if audio.ndim == 1:
            return audio[np.newaxis, :]

        # 2D 数组需要推断
        if audio.ndim == 2:
            dim0, dim1 = audio.shape

            # 如果第一维是常见采样率，第二维是合理的通道数
            if dim0 in self.COMMON_SAMPLE_RATES and dim1 <= 8:
                # 这种情况很可能是搞反了
                if dim0 % sample_rate == 0 or dim0 > sample_rate:
                    return audio.T

            # 如果第二维是常见采样率，第一维是合理的通道数
            if dim1 in self.COMMON_SAMPLE_RATES and dim0 <= 8:
                return audio

            # 如果第一维远大于第二维，可能已经是 channel_first
            if dim0 > dim1 * 2:
                return audio

            # 如果第二维远大于第一维，可能是 (samples, channels)
            if dim1 > dim0 * 2:
                return audio.T

        # 保守策略：默认返回原数组（假设已经是 channel_first）
        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """重采样"""
        try:
            import librosa
            if audio.ndim == 2:
                result = np.zeros((audio.shape[0], int(audio.shape[1] * target_sr / orig_sr)))
                for ch in range(audio.shape[0]):
                    result[ch] = librosa.resample(
                        audio[ch],
                        orig_sr=orig_sr,
                        target_sr=target_sr
                    )
                return result
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

    def load_segment(
        self,
        file_path: str,
        start: float,
        end: Optional[float] = None,
        force_channel_first: Optional[bool] = None
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频片段

        Args:
            file_path: 音频文件路径
            start: 起始时间(秒)
            end: 结束时间(秒)
            force_channel_first: 强制指定输出格式

        Returns:
            (audio_data, sample_rate)
        """
        audio, sr = self.load(file_path, force_channel_first=True)

        start_sample = int(start * sr)
        if end is not None:
            end_sample = int(end * sr)
            audio = audio[:, start_sample:end_sample]
        else:
            audio = audio[:, start_sample:]

        return audio, sr


class AudioSaver:
    """
    音频保存器

    支持多种音频格式的保存。
    """

    def __init__(
        self,
        normalize: bool = False,
        target_db: float = -3.0
    ):
        """
        初始化保存器

        Args:
            normalize: 是否归一化
            target_db: 目标分贝值
        """
        self.normalize = normalize
        self.target_db = target_db

    def save(
        self,
        audio: np.ndarray,
        file_path: str,
        sample_rate: int = 44100,
        format: Optional[str] = None,
        bit_depth: Optional[int] = 16,
        force_channel_first: Optional[bool] = None
    ) -> bool:
        """
        保存音频文件

        Args:
            audio: 音频数据
            file_path: 保存路径
            sample_rate: 采样率
            format: 音频格式
            bit_depth: 位深
            force_channel_first: 指定输入格式

        Returns:
            bool: 是否成功
        """
        # 准备保存路径
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = path.suffix[1:].lower()

        # 准备音频数据（转换为 soundfile 需要的 (samples, channels) 格式）
        audio = self._prepare_audio(audio, force_channel_first)

        try:
            import soundfile as sf

            # 确定 subtype
            subtype = self._get_subtype(format, bit_depth)

            # 写入文件
            sf.write(
                str(path),
                audio,
                sample_rate,
                format=format.upper(),
                subtype=subtype
            )

            return True

        except ImportError:
            # 降级到 pydub
            return self._save_with_pydub(audio, path, sample_rate, format)

    def _prepare_audio(
        self,
        audio: np.ndarray,
        force_channel_first: Optional[bool] = None
    ) -> np.ndarray:
        """
        准备音频数据，转换为 (samples, channels) 格式

        Args:
            audio: 输入音频
            force_channel_first: 指定输入格式

        Returns:
            (samples, channels) 格式的音频
        """
        # 确保是 2D 数组
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # 确定是否需要转置
        needs_transpose = False

        if force_channel_first is not None:
            # 显式指定
            needs_transpose = force_channel_first
        elif self._looks_like_channel_first(audio):
            # 检测是否看起来像 channel_first
            needs_transpose = True

        if needs_transpose and audio.shape[0] < audio.shape[1]:
            audio = audio.T

        # 归一化
        if self.normalize:
            audio = self._normalize(audio)

        # 限制范围
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def _looks_like_channel_first(self, audio: np.ndarray) -> bool:
        """
        检测音频是否可能是 channel_first 格式

        检测方法：
        1. 如果第一维 <= 8，很可能是通道数
        2. 如果第二维是采样率的倍数，很可能是样本数
        """
        if audio.ndim != 2:
            return False

        channels, samples = audio.shape

        # 通道数通常 <= 8
        if channels > 8:
            return False

        # 如果第一维是常见的采样率，可能是搞反了
        COMMON_SAMPLE_RATES = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000}
        if channels in COMMON_SAMPLE_RATES and samples <= 8:
            return True

        return False

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """归一化音频"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_linear = 10 ** (self.target_db / 20)
            audio = audio * (target_linear / max_val)
        return audio

    def _get_subtype(self, format: str, bit_depth: int) -> str:
        """获取音频子类型"""
        lossless = {"wav", "flac", "aiff"}
        lossy = {"mp3", "ogg", "aac", "m4a"}

        if format.lower() in lossless:
            bit_map = {
                16: "PCM_16",
                24: "PCM_24",
                32: "PCM_32",
            }
        elif format.lower() in lossy:
            return "VORBIS" if format.lower() == "ogg" else "AAC"
        else:
            return "PCM_16"

        return bit_map.get(bit_depth, "PCM_16")

    def _save_with_pydub(
        self,
        audio: np.ndarray,
        path: Path,
        sample_rate: int,
        format: str
    ) -> bool:
        """使用 pydub 保存"""
        from pydub import AudioSegment

        # 确保是 (samples, channels) 格式
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T

        channels = audio.shape[1] if audio.ndim > 1 else 1

        # 转换回 int16
        audio_int = (audio * 32767).astype(np.int16)

        # 创建 AudioSegment
        segment = AudioSegment(
            audio_int.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=channels,
        )

        segment.export(str(path), format=format)
        return True

    def save_tracks(
        self,
        tracks: dict,
        output_dir: str,
        sample_rate: int = 44100,
        format: str = "wav",
    ) -> dict:
        """
        保存多个音轨

        Args:
            tracks: 音轨字典 {name: audio}
            output_dir: 输出目录
            sample_rate: 采样率
            format: 音频格式

        Returns:
            dict: 保存结果 {name: path}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        for name, audio in tracks.items():
            file_path = output_path / f"{name}.{format}"
            self.save(audio, str(file_path), sample_rate, format)
            saved_paths[name] = str(file_path)

        return saved_paths
