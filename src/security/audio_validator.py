"""
SOMA 安全模块 - 音频文件验证

提供音频文件安全检查，包括：
- 文件格式验证
- 采样率范围检查
- 时长限制检查
- 文件大小检查
- MIME 类型检测
- 音频元数据验证

使用方式:
    from src.security.audio_validator import AudioValidator, validate_audio

    # 创建验证器
    validator = AudioValidator()

    # 验证音频文件
    result = validator.validate("/path/to/audio.wav")

    # 使用便捷函数
    result = validate_audio("/path/to/audio.wav", max_duration=300)
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from src.config import AudioUtilsDefaults
from src.security.path_validator import safe_path
from src.exceptions import AudioFormatError as BaseAudioFormatError
from src.exceptions import AudioProcessingError as BaseAudioProcessingError

logger = logging.getLogger(__name__)


class AudioFormatError(BaseAudioFormatError):
    """音频格式错误异常"""
    pass


class AudioValidationError(BaseAudioProcessingError):
    """音频验证失败异常"""
    pass


class AudioFormat(Enum):
    """支持的音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    AAC = "aac"
    WMA = "wma"
    AIFF = "aiff"
    UNKNOWN = "unknown"


@dataclass
class AudioMetadata:
    """音频元数据"""
    format: AudioFormat
    sample_rate: int
    channels: int
    bit_depth: Optional[int]
    duration: float  # 秒
    bitrate: Optional[int]  # bps
    file_size: int  # 字节
    codec: Optional[str] = None
    is_lossless: bool = False


@dataclass
class AudioValidationResult:
    """音频验证结果"""
    is_valid: bool
    metadata: Optional[AudioMetadata] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AudioValidator:
    """
    音频文件验证器

    功能:
    - 验证文件是否为有效的音频文件
    - 检查文件格式是否支持
    - 验证采样率范围
    - 检查时长限制
    - 验证文件大小
    - 读取音频元数据

    使用示例:
        validator = AudioValidator(
            allowed_formats=[AudioFormat.WAV, AudioFormat.MP3],
            max_duration=3600,
            max_file_size_mb=500
        )

        result = validator.validate("/path/to/audio.wav")
        if result.is_valid:
            print(f"采样率: {result.metadata.sample_rate}")
        else:
            print(f"验证失败: {result.errors}")
    """

    # 常见音频文件魔数
    MAGIC_BYTES: Dict[AudioFormat, bytes] = {
        AudioFormat.WAV: b'RIFF',
        AudioFormat.FLAC: b'fLaC',
        AudioFormat.OGG: b'OggS',
        AudioFormat.MP3: b'\xff\xfb',  # MP3 frame sync
        AudioFormat.M4A: b'ftyp',       # MP4/M4A
    }

    # MIME 类型映射
    MIME_TYPES: Dict[str, AudioFormat] = {
        'audio/wav': AudioFormat.WAV,
        'audio/x-wav': AudioFormat.WAV,
        'audio/mpeg': AudioFormat.MP3,
        'audio/mp3': AudioFormat.MP3,
        'audio/flac': AudioFormat.FLAC,
        'audio/ogg': AudioFormat.OGG,
        'audio/x-m4a': AudioFormat.M4A,
        'audio/aac': AudioFormat.AAC,
    }

    def __init__(
        self,
        allowed_formats: Optional[List[AudioFormat]] = None,
        min_sample_rate: int = 8000,
        max_sample_rate: int = 192000,
        min_duration: float = 0.1,  # 最小 100ms
        max_duration: float = 3600,  # 最大 1 小时
        max_file_size_mb: int = 500,
        defaults: Optional[AudioUtilsDefaults] = None
    ):
        """
        初始化音频验证器

        Args:
            allowed_formats: 允许的音频格式列表
            min_sample_rate: 最小采样率
            max_sample_rate: 最大采样率
            min_duration: 最小音频时长（秒）
            max_duration: 最大音频时长（秒）
            max_file_size_mb: 最大文件大小（MB）
            defaults: 音频工具默认配置
        """
        self.defaults = defaults or AudioUtilsDefaults()

        # 使用配置或参数
        self.allowed_formats = allowed_formats or [
            AudioFormat(f) for f in self.defaults.allowed_audio_formats
        ] if hasattr(self.defaults, 'allowed_audio_formats') else [
            AudioFormat.WAV, AudioFormat.MP3, AudioFormat.FLAC,
            AudioFormat.OGG, AudioFormat.M4A
        ]

        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration or self.defaults.max_duration_seconds
        self.max_file_size_bytes = (max_file_size_mb or self.defaults.max_file_size_mb) * 1024 * 1024
        self.min_file_size_bytes = self.defaults.min_file_size_bytes

    def validate(
        self,
        path: Union[str, Path],
        check_metadata: bool = True
    ) -> AudioValidationResult:
        """
        验证音频文件

        Args:
            path: 音频文件路径
            check_metadata: 是否读取元数据进行验证

        Returns:
            AudioValidationResult 验证结果
        """
        result = AudioValidationResult(is_valid=False)
        
        # 0. 先转换为 Path 对象，检查文件是否存在
        # 不存在的文件返回 is_valid=False，而不是抛出异常
        if isinstance(path, str):
            path = Path(path)
        
        # 1. 检查文件是否存在
        if not path.exists():
            result.errors.append(f"文件不存在: {path}")
            return result
        
        # 2. 检查路径安全性（仅对存在的文件）
        try:
            path = safe_path(path)
        except Exception as e:
            result.errors.append(f"路径安全检查失败: {str(e)}")
            return result

        # 2. 检查文件大小
        self._check_file_size(path, result)
        if result.errors:
            return result

        # 3. 检测文件格式
        detected_format = self._detect_format(path)
        result.metadata = AudioMetadata(
            format=detected_format,
            sample_rate=0,
            channels=0,
            bit_depth=None,
            duration=0.0,
            bitrate=None,
            file_size=path.stat().st_size
        )

        # 4. 检查格式是否允许
        if not self._is_format_allowed(detected_format):
            result.errors.append(
                f"不支持的音频格式: {detected_format.value}，"
                f"允许的格式: {[f.value for f in self.allowed_formats]}"
            )
            return result

        # 5. 读取元数据（如果需要）
        if check_metadata:
            self._check_metadata(path, result)

        if not result.errors:
            result.is_valid = True

        return result

    def _check_file_size(
        self,
        path: Path,
        result: AudioValidationResult
    ) -> None:
        """检查文件大小"""
        file_size = path.stat().st_size

        if file_size < self.min_file_size_bytes:
            result.errors.append(
                f"文件太小 ({file_size} bytes)，最小要求: {self.min_file_size_bytes} bytes"
            )

        if file_size > self.max_file_size_bytes:
            result.errors.append(
                f"文件太大 ({file_size / 1024 / 1024:.1f} MB)，"
                f"最大允许: {self.max_file_size_bytes / 1024 / 1024:.1f} MB"
            )

    def _detect_format(self, path: Path) -> AudioFormat:
        """检测音频文件格式"""
        # 1. 通过扩展名猜测
        ext = path.suffix.lower().lstrip('.')

        # 特殊处理
        if ext in ['mp2', 'mp3']:
            ext = 'mp3'
        elif ext in ['aif', 'aifc']:
            ext = 'aiff'

        try:
            return AudioFormat(ext)
        except ValueError:
            pass

        # 2. 通过魔数检测
        try:
            with open(path, 'rb') as f:
                header = f.read(16)

                for fmt, magic in self.MAGIC_BYTES.items():
                    if header.startswith(magic):
                        return fmt

                # MP3 特殊检测（可能有 ID3 头）
                if len(header) >= 3 and header[0:2] in [b'\xff\xfb', b'\xff\xf3', b'\xff\xf2']:
                    return AudioFormat.MP3

        except Exception as e:
            logger.warning(f"无法读取文件头: {e}")

        return AudioFormat.UNKNOWN

    def _is_format_allowed(self, fmt: AudioFormat) -> bool:
        """检查格式是否允许"""
        if fmt == AudioFormat.UNKNOWN:
            return False
        return fmt in self.allowed_formats

    def _check_metadata(
        self,
        path: Path,
        result: AudioValidationResult
    ) -> None:
        """检查音频元数据"""
        try:
            import soundfile as sf

            # 读取音频信息
            info = sf.info(str(path))

            # 更新元数据
            result.metadata.sample_rate = info.samplerate
            result.metadata.channels = info.channels
            result.metadata.duration = info.duration
            result.metadata.codec = info.format

            # 检测是否为无损格式
            result.metadata.is_lossless = info.format.lower() in ['wav', 'flac', 'aiff']

            # 获取子格式信息
            if hasattr(info, 'subtype'):
                if info.subtype in ['PCM_16', 'PCM_24', 'PCM_32']:
                    result.metadata.bit_depth = int(info.subtype.split('_')[1])

            # 检查采样率
            if result.metadata.sample_rate < self.min_sample_rate:
                result.errors.append(
                    f"采样率过低 ({result.metadata.sample_rate} Hz)，"
                    f"最小要求: {self.min_sample_rate} Hz"
                )
            elif result.metadata.sample_rate > self.max_sample_rate:
                result.warnings.append(
                    f"采样率较高 ({result.metadata.sample_rate} Hz)，"
                    f"可能影响处理速度"
                )

            # 检查时长
            if result.metadata.duration < self.min_duration:
                result.errors.append(
                    f"音频时长过短 ({result.metadata.duration:.2f}s)，"
                    f"最小要求: {self.min_duration}s"
                )
            elif result.metadata.duration > self.max_duration:
                result.errors.append(
                    f"音频时长过长 ({result.metadata.duration:.1f}s)，"
                    f"最大允许: {self.max_duration}s"
                )

            # 检查声道数
            if result.metadata.channels > 2:
                result.warnings.append(
                    f"音频为 {result.metadata.channels} 声道，将被转换为立体声"
                )

        except ImportError:
            result.warnings.append("soundfile 未安装，无法验证音频元数据")
        except Exception as e:
            result.errors.append(f"无法读取音频元数据: {e}")

    def get_metadata(
        self,
        path: Union[str, Path]
    ) -> Optional[AudioMetadata]:
        """
        获取音频元数据

        Args:
            path: 音频文件路径

        Returns:
            AudioMetadata 或 None
        """
        result = self.validate(path, check_metadata=True)
        return result.metadata if result.is_valid else None


# 全局验证器实例
_default_validator: Optional[AudioValidator] = None


def get_audio_validator() -> AudioValidator:
    """获取全局音频验证器实例"""
    global _default_validator
    if _default_validator is None:
        _default_validator = AudioValidator()
    return _default_validator


def validate_audio(
    path: Union[str, Path],
    max_duration: Optional[float] = None,
    check_metadata: bool = True
) -> AudioValidationResult:
    """
    便捷的音频验证函数

    Args:
        path: 音频文件路径
        max_duration: 最大时长限制（秒）
        check_metadata: 是否检查元数据

    Returns:
        AudioValidationResult

    示例:
        result = validate_audio("/path/to/audio.wav")
        if result.is_valid:
            print(f"音频有效，采样率: {result.metadata.sample_rate}")
    """
    validator = get_audio_validator()

    if max_duration is not None:
        # 临时设置最大时长
        original_max = validator.max_duration
        validator.max_duration = max_duration
        try:
            return validator.validate(path, check_metadata)
        finally:
            validator.max_duration = original_max

    return validator.validate(path, check_metadata)
