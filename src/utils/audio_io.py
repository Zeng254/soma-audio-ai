"""
Audio I/O - 音频读写工具
提供统一的音频文件读写接口
"""

from pathlib import Path
from typing import Tuple, Optional, Union, List
import numpy as np


class AudioLoader:
    """
    音频加载器
    
    支持多种音频格式的加载和预处理
    """
    
    SUPPORTED_FORMATS = {
        "wav", "mp3", "flac", "ogg", "aac", 
        "m4a", "wma", "aiff", "amr", "opus"
    }
    
    def __init__(self, target_sr: Optional[int] = None, mono: bool = False):
        """
        初始化加载器
        
        Args:
            target_sr: 目标采样率，None 表示保持原采样率
            mono: 是否转换为单声道
        """
        self.target_sr = target_sr
        self.mono = mono
    
    def load(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            (audio_data, sample_rate)
        """
        path = Path(file_path)
        suffix = path.suffix[1:].lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}")
        
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype='float32')
            
        except ImportError:
            # 降级到 pydub
            from pydub import AudioSegment
            audio_segment = AudioSegment.from_file(str(path))
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            sr = audio_segment.frame_rate
            
            # 转换为立体声格式
            if audio_segment.channels == 2:
                audio = audio.reshape((-1, 2)).T
            else:
                audio = audio[np.newaxis, :]
        
        # 归一化到 [-1, 1]
        audio = audio / np.iinfo(np.int16).max if audio.dtype == np.int16 else audio
        
        # 转换单声道
        if self.mono and audio.shape[0] > 1:
            audio = np.mean(audio, axis=0, keepdims=True)
        
        # 确保格式为 (channels, samples)
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T
        
        # 重采样
        if self.target_sr and self.target_sr != sr:
            audio = self._resample(audio, sr, self.target_sr)
            sr = self.target_sr
        
        return audio, sr
    
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
            num_samples = int(len(audio[0]) * target_sr / orig_sr)
            return signal.resample(audio, num_samples, axis=-1)
    
    def load_segment(
        self,
        file_path: str,
        start: float,
        end: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频片段
        
        Args:
            file_path: 音频文件路径
            start: 起始时间(秒)
            end: 结束时间(秒)
            
        Returns:
            (audio_data, sample_rate)
        """
        audio, sr = self.load(file_path)
        
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
    
    支持多种音频格式的保存
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
    ) -> bool:
        """
        保存音频文件
        
        Args:
            audio: 音频数据
            file_path: 保存路径
            sample_rate: 采样率
            format: 音频格式
            bit_depth: 位深
            
        Returns:
            bool: 是否成功
        """
        # 准备保存路径
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format is None:
            format = path.suffix[1:].lower()
        
        # 准备音频数据
        audio = self._prepare_audio(audio)
        
        try:
            import soundfile as sf
            
            # 确定 subtype
            subtype = self._get_subtype(format, bit_depth)
            
            # 写入文件
            # soundfile 需要 (samples, channels) 格式
            if audio.shape[0] < audio.shape[1]:
                sf.write(str(path), audio.T, sample_rate, format=format.upper(), subtype=subtype)
            else:
                sf.write(str(path), audio.T, sample_rate, format=format.upper(), subtype=subtype)
            
            return True
            
        except ImportError:
            # 降级到 pydub
            return self._save_with_pydub(audio, path, sample_rate, format)
    
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """准备音频数据"""
        # 确保是 2D 数组
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        
        # 确保格式为 (samples, channels)
        if audio.shape[0] > audio.shape[1]:
            audio = audio.T
        
        # 归一化
        if self.normalize:
            audio = self._normalize(audio)
        
        # 限制范围
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
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
        
        # 转换回 int16
        audio_int = (audio * 32767).astype(np.int16)
        
        if audio_int.shape[0] > audio_int.shape[1]:
            audio_int = audio_int.T
        
        # 创建 AudioSegment
        segment = AudioSegment(
            audio_int.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=audio_int.shape[1] if audio_int.ndim > 1 else 1,
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
        
        results = {}
        for name, audio in tracks.items():
            file_path = str(output_path / f"{name}.{format}")
            success = self.save(audio, file_path, sample_rate, format)
            results[name] = file_path if success else None
        
        return results


class AudioStreamer:
    """
    音频流处理器
    
    支持大文件的流式读取和处理
    """
    
    def __init__(self, chunk_size: int = 8192):
        """
        初始化流处理器
        
        Args:
            chunk_size: 块大小(采样点)
        """
        self.chunk_size = chunk_size
    
    def stream_chunks(
        self,
        file_path: str,
        process_fn: Optional[callable] = None,
    ):
        """
        流式读取音频块
        
        Args:
            file_path: 音频文件路径
            process_fn: 处理函数，接收 (chunk, sample_rate)
            
        Yields:
            处理后的音频块
        """
        try:
            import soundfile as sf
            info = sf.info(file_path)
            sr = info.samplerate
            
            with sf.SoundFile(file_path) as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if len(chunk) == 0:
                        break
                    
                    if process_fn:
                        chunk = process_fn(chunk, sr)
                    
                    yield chunk, sr
                    
        except ImportError:
            from pydub import AudioSegment
            segment = AudioSegment.from_file(file_path)
            sr = segment.frame_rate
            
            # 逐块处理
            for i in range(0, len(segment), self.chunk_size):
                chunk = segment[i:i + self.chunk_size]
                chunk_array = np.array(chunk.get_array_of_samples(), dtype=np.float32)
                chunk_array = chunk_array / 32768.0
                
                if chunk.channels == 2:
                    chunk_array = chunk_array.reshape((-1, 2)).T
                else:
                    chunk_array = chunk_array[np.newaxis, :]
                
                if process_fn:
                    chunk_array = process_fn(chunk_array, sr)
                
                yield chunk_array, sr
