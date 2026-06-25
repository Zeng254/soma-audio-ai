"""
Audio Converter - 音频格式转换器
支持多种音频格式之间的转换
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List
import numpy as np

import ffmpeg


class ConversionFormat(Enum):
    """支持的音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    AAC = "aac"
    OGG = "ogg"
    M4A = "m4a"
    WMA = "wma"
    AIFF = "aiff"
    AMR = "amr"
    
    # 无损格式
    LOSSLESS_FORMATS = {WAV, FLAC, AIFF}
    
    # 有损格式
    LOSSY_FORMATS = {MP3, AAC, OGG, M4A, WMA, AMR}


@dataclass
class AudioMetadata:
    """音频元数据"""
    format: str
    sample_rate: int
    channels: int
    bit_rate: Optional[int] = None
    duration: Optional[float] = None
    artist: Optional[str] = None
    title: Optional[str] = None
    album: Optional[str] = None
    year: Optional[int] = None
    genre: Optional[str] = None


class AudioConverter:
    """
    音频格式转换器
    
    基于 FFmpeg 实现高质量音频格式转换。
    
    支持的转换:
    - 格式转换 (MP3 -> WAV, FLAC -> MP3, etc.)
    - 采样率转换 (44100 -> 48000)
    - 声道转换 (Stereo -> Mono)
    - 比特率调整
    - 质量预设
    """
    
    # 质量预设
    QUALITY_PRESETS = {
        "ultra": {"codec": "libFLAC", "compression": 0},
        "high": {"codec": "libFLAC", "compression": 3},
        "medium": {"codec": "libmp3lame", "qscale": 2},
        "low": {"codec": "libmp3lame", "qscale": 4},
    }
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        初始化转换器
        
        Args:
            ffmpeg_path: FFmpeg 可执行文件路径
        """
        self.ffmpeg_path = ffmpeg_path
    
    def convert(
        self,
        input_path: str,
        output_path: str,
        output_format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        bit_rate: Optional[str] = None,
        quality: str = "high",
        **kwargs
    ) -> bool:
        """
        转换音频文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            output_format: 输出格式 (wav, mp3, flac, etc.)
            sample_rate: 目标采样率
            channels: 目标声道数
            bit_rate: 目标比特率
            quality: 质量预设
            **kwargs: FFmpeg 其他参数
            
        Returns:
            bool: 转换是否成功
        """
        # 确定输出格式
        if output_format is None:
            output_format = Path(output_path).suffix[1:].lower()
        
        try:
            # 构建 FFmpeg 命令
            stream = ffmpeg.input(input_path)
            
            # 音频过滤器
            filters = []
            if sample_rate:
                filters.append(f"aformat=sample_fmts=fltp:sample_rates={sample_rate}")
            if channels:
                if channels == 1:
                    filters.append("aformat=channel_layouts=mono")
                elif channels == 2:
                    filters.append("aformat=channel_layouts=stereo")
            
            if filters:
                stream = ffmpeg.filter(stream, 'afilter', ','.join(filters))
            
            # 获取编码器设置
            codec_settings = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS["high"])
            
            # 构建输出参数
            output_kwargs = {}
            if "codec" in codec_settings:
                output_kwargs["acodec"] = codec_settings["codec"]
            if "qscale" in codec_settings:
                output_kwargs["aq"] = codec_settings["qscale"]
            if bit_rate:
                output_kwargs["audio_bitrate"] = bit_rate
            
            # 添加额外参数
            output_kwargs.update(kwargs)
            
            # 执行转换
            ffmpeg.output(stream, output_path, **output_kwargs).run(
                cmd=self.ffmpeg_path,
                overwrite_output=True,
                quiet=True,
            )
            
            return True
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return False
    
    def convert_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        output_format: str = "wav",
        **kwargs
    ) -> bool:
        """
        将音频数组转换为文件
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            output_path: 输出文件路径
            output_format: 输出格式
            **kwargs: 其他参数
            
        Returns:
            bool: 转换是否成功
        """
        try:
            import soundfile as sf
            
            # 确保音频格式正确
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            elif audio.shape[0] > audio.shape[1]:
                audio = audio.T
            
            # 写入文件
            sf.write(output_path, audio.T, sample_rate, format=output_format.upper())
            return True
            
        except ImportError:
            print("soundfile not installed. Use convert() for file conversion.")
            return False
        except Exception as e:
            print(f"Error writing audio: {e}")
            return False
    
    def get_metadata(self, file_path: str) -> Optional[AudioMetadata]:
        """
        获取音频元数据
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            AudioMetadata: 元数据对象
        """
        try:
            probe = ffmpeg.probe(file_path, cmd=self.ffmpeg_path)
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"),
                None
            )
            
            if audio_stream is None:
                return None
            
            format_info = probe["format"]
            
            # 解析标签
            tags = audio_stream.get("tags", {})
            
            return AudioMetadata(
                format=format_info.get("format_name", "unknown"),
                sample_rate=int(audio_stream.get("sample_rate", 44100)),
                channels=int(audio_stream.get("channels", 2)),
                bit_rate=int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None,
                duration=float(format_info.get("duration", 0)),
                artist=tags.get("artist"),
                title=tags.get("title"),
                album=tags.get("album"),
                year=int(tags.get("date")) if tags.get("date") else None,
                genre=tags.get("genre"),
            )
            
        except ffmpeg.Error:
            return None
    
    def batch_convert(
        self,
        input_files: List[str],
        output_dir: str,
        output_format: str,
        **kwargs
    ) -> dict:
        """
        批量转换
        
        Args:
            input_files: 输入文件列表
            output_dir: 输出目录
            output_format: 输出格式
            **kwargs: 转换参数
            
        Returns:
            dict: 转换结果 {file: success}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for input_file in input_files:
            input_name = Path(input_file).stem
            output_file = str(output_path / f"{input_name}.{output_format}")
            results[input_file] = self.convert(input_file, output_file, output_format, **kwargs)
        
        return results
    
    def normalize_audio(
        self,
        input_path: str,
        output_path: str,
        target_db: float = -20.0,
    ) -> bool:
        """
        归一化音频电平
        
        Args:
            input_path: 输入文件
            output_path: 输出文件
            target_db: 目标分贝值
            
        Returns:
            bool: 是否成功
        """
        try:
            stream = ffmpeg.input(input_path)
            
            # loudnorm 滤波器
            filtered = ffmpeg.filter(
                stream,
                "loudnorm",
                I=str(target_db),
                TP=-1.5,
                LRA=11,
            )
            
            ffmpeg.output(filtered, output_path, overwrite_output=True).run(
                cmd=self.ffmpeg_path,
                quiet=True,
            )
            return True
            
        except ffmpeg.Error:
            return False
    
    def trim_audio(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: Optional[float] = None,
    ) -> bool:
        """
        裁剪音频
        
        Args:
            input_path: 输入文件
            output_path: 输出文件
            start_time: 起始时间(秒)
            end_time: 结束时间(秒)
            
        Returns:
            bool: 是否成功
        """
        try:
            if end_time:
                stream = ffmpeg.input(input_path, ss=start_time, to=end_time)
            else:
                stream = ffmpeg.input(input_path, ss=start_time)
            
            ffmpeg.output(stream, output_path, overwrite_output=True).run(
                cmd=self.ffmpeg_path,
                quiet=True,
            )
            return True
            
        except ffmpeg.Error:
            return False
    
    def merge_audio(
        self,
        input_files: List[str],
        output_path: str,
        crossfade: float = 0.0,
    ) -> bool:
        """
        合并多个音频文件
        
        Args:
            input_files: 输入文件列表
            output_path: 输出文件
            crossfade: 交叉淡入淡出时长
            
        Returns:
            bool: 是否成功
        """
        if not input_files:
            return False
        
        try:
            if len(input_files) == 1:
                # 单个文件，直接复制
                import shutil
                shutil.copy(input_files[0], output_path)
                return True
            
            # 复杂合并使用 filter_complex
            inputs = [ffmpeg.input(f) for f in input_files]
            
            if crossfade > 0:
                # 带交叉淡入淡出的合并
                merged = inputs[0]
                for inp in inputs[1:]:
                    merged = ffmpeg.filter(
                        [merged, inp],
                        "acrossfade",
                        d=crossfade,
                    )
            else:
                # 直接拼接
                merged = ffmpeg.concat(*inputs, a=1)
            
            ffmpeg.output(merged, output_path, overwrite_output=True).run(
                cmd=self.ffmpeg_path,
                quiet=True,
            )
            return True
            
        except ffmpeg.Error:
            return False
