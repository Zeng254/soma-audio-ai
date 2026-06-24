"""
SOMA 配置中心 - 默认配置参数

本模块集中管理所有模块的默认配置参数，包括：
- 分离器 (Separators)
- 声音转换器 (Voice Converters)
- 音效处理器 (Effects)
- 格式转换器 (Converters)
- 音频工具 (Audio Utils)
- 安全设置 (Security)
- 日志设置 (Logging)

所有硬编码的默认值都应迁移到此模块。
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class SeparatorDefaults:
    """分离器默认配置"""
    # 模型相关
    default_model: str = "htdemucs_ft"
    models_cache_dir: str = "~/.cache/torch/hub/demucs"
    device: str = "auto"  # auto, cpu, cuda, mps

    # 处理参数
    segment_size: int = 0  # 0 表示不分割，-1 表示自动
    overlap: float = 0.5  # 重叠率 0-1
    batch_size: int = 1
    num_workers: int = 0  # 0 表示使用主进程

    # 输出相关
    output_format: str = "wav"  # wav, mp3, flac, ogg
    output_bit_depth: int = 32  # 16, 24, 32
    output_sample_rate: int = 44100

    # 特定模型参数
    stem_types: list = field(default_factory=lambda: ["vocals", "drums", "bass", "other"])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


@dataclass
class VoiceConverterDefaults:
    """声音转换器默认配置"""
    # 引擎相关
    default_engine: str = "auto"  # auto, rvc, sovits
    device: str = "auto"

    # 通用处理参数
    pitch_shift: float = 0.0  # 半音调整
    pitch_algo: str = "rmvpe"  # rmvpe, crepe, dio, harvest, pm
    vpm: float = 0.5  # 音色匹配 0-1
    rms_mix: float = 0.5  # 响度混合 0-1

    # 质量参数
    f0_smooth: int = 0  # 音高平滑帧数
    protect: float = 0.33  # 保护非语音部分

    # SoVITS 特有参数
    sovits_version: str = "4.1"
    diffusion_steps: int = 10  # 扩散步数
    diffusion_seed: int = 42

    # RVC 特有参数
    rvc_version: str = "v2"
    index_rate: float = 0.3  # 索引搜索比率
    filter_radius: int = 3  # 滤波半径

    # 音频参数
    input_sample_rate: int = 44100
    output_sample_rate: int = 44100
    hop_length: int = 128

    # 模型缓存
    models_cache_dir: str = "~/.soma/models"

    def to_dict(self) -> dict:
        """转换为字典用于 JSON 序列化"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                result[key] = value
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = str(value)
        return result


@dataclass
class EffectsDefaults:
    """音效处理器默认配置"""

    @dataclass
    class EQDefaults:
        """均衡器默认配置"""
        enabled: bool = True
        sample_rate: int = 44100
        bands: int = 10  # 频段数

        # 频段配置 (10段)
        band_frequencies: list = field(default_factory=lambda: [
            31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000
        ])

        # 默认增益 (dB)
        default_gains: list = field(default_factory=lambda: [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])

        # 预设类型
        preset: str = "flat"  # flat, vocal_enhance, bass_boost, treble_boost

        def to_dict(self) -> dict:
            """转换为字典用于 JSON 序列化"""
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith('_')}

    @dataclass
    class ReverbDefaults:
        """混响效果默认配置"""
        enabled: bool = True
        sample_rate: int = 44100

        # 混响类型
        reverb_type: str = "room"  # room, hall, plate, cathedral

        # 房间参数
        room_size: float = 0.5  # 0-1
        damping: float = 0.5  # 0-1
        wet_level: float = 0.3  # 0-1
        dry_level: float = 0.7  # 0-1
        width: float = 1.0  # 0-1
        freeze_mode: float = 0.0  # 0-1

        # 预设
        preset: str = "natural"  # natural, dramatic, subtle, wet

        def to_dict(self) -> dict:
            """转换为字典用于 JSON 序列化"""
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith('_')}

    @dataclass
    class PitchDefaults:
        """音调变换默认配置"""
        enabled: bool = True
        sample_rate: int = 44100

        # 音调参数
        semitones: float = 0.0  # 半音调整
        cents: float = 0.0  # 音分调整
        formants_shift: float = 0.0  # 共振峰调整

        # 算法参数
        algorithm: str = "soundtouch"  # soundtouch, librosa, pitchy
        quick: bool = False  # 快速模式
        fft_size: int = 2048

        def to_dict(self) -> dict:
            """转换为字典用于 JSON 序列化"""
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith('_')}

    eq: EQDefaults = field(default_factory=EQDefaults)
    reverb: ReverbDefaults = field(default_factory=ReverbDefaults)
    pitch: PitchDefaults = field(default_factory=PitchDefaults)

    def to_dict(self) -> dict:
        """转换为字典用于 JSON 序列化"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                result[key] = value
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = str(value)
        return result


@dataclass
class ConverterDefaults:
    """格式转换器默认配置"""
    # 输入参数
    input_format: str = "auto"  # auto 检测格式

    # 输出参数
    output_format: str = "wav"
    output_bit_depth: int = 16  # 8, 16, 24, 32
    output_sample_rate: int = 44100
    output_channels: int = 2  # 1 单声道, 2 立体声

    # 编码参数
    codec: str = "pcm_s16le"  # libmp3lame, aac, opus, pcm_s16le
    bitrate: str = "192k"  # 128k, 192k, 320k
    quality: int = 5  # 0-9 (VBR), 0 最好

    # ffmpeg 参数
    ffmpeg_path: str = "ffmpeg"
    timeout: int = 300  # 秒

    def to_dict(self) -> dict:
        """转换为字典用于 JSON 序列化"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class AudioUtilsDefaults:
    """音频工具默认配置"""
    # 音频参数
    default_sample_rate: int = 44100
    default_channels: int = 2
    default_bit_depth: int = 16

    # 文件大小限制
    min_file_size_bytes: int = 1024  # 最小文件大小 1KB
    max_file_size_mb: int = 500  # 500MB
    max_duration_seconds: int = 3600  # 1小时

    # 缓存设置
    cache_enabled: bool = True
    cache_dir: str = "~/.soma/cache"
    cache_max_size_mb: int = 1024  # 1GB

    # 临时文件
    temp_dir: str = "~/.soma/temp"
    auto_cleanup: bool = True

    def to_dict(self) -> dict:
        """转换为字典用于 JSON 序列化"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class SecurityDefaults:
    """安全设置默认配置"""
    # 路径限制
    allowed_base_dirs: list = field(default_factory=lambda: [
        "~/.soma/workspace",
        "~/Documents",
        "~/Music",
        "/tmp"
    ])
    allow_symlinks: bool = False
    max_path_depth: int = 20

    # 文件验证
    allowed_audio_formats: list = field(default_factory=lambda: [
        "wav", "mp3", "flac", "ogg", "m4a", "aac", "wma"
    ])
    max_file_size_mb: int = 500
    min_file_size_bytes: int = 1024  # 1KB

    # 模型验证
    max_model_size_mb: int = 5000  # 5GB
    trusted_model_signatures: list = field(default_factory=list)

    # 网络安全
    allow_network_access: bool = True  # 是否允许网络访问
    allowed_hosts: list = field(default_factory=lambda: [
        "huggingface.co",
        "github.com"
    ])

    def to_dict(self) -> dict:
        """转换为字典用于 JSON 序列化"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class LoggingDefaults:
    """日志系统默认配置"""
    # 日志级别
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # 日志输出
    console_output: bool = True
    file_output: bool = True

    # 日志文件
    log_dir: str = "~/.soma/logs"
    log_file_name: str = "soma.log"
    max_log_size_mb: int = 10
    backup_count: int = 7  # 保留7天

    # 日志格式
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # 模块级别日志
    module_levels: Dict[str, str] = field(default_factory=lambda: {
        "soma": "INFO",
        "soma.separators": "INFO",
        "soma.voice_converters": "INFO",
        "soma.effects": "INFO",
        "soma.pipeline": "DEBUG"
    })

    def to_dict(self) -> dict:
        """转换为字典用于 JSON 序列化"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class SomaDefaults:
    """SOMA 主配置"""
    # 版本信息
    app_name: str = "SOMA"
    version: str = "0.1.0"

    # 子配置 (复数形式，与 Config API 保持一致)
    separators: SeparatorDefaults = field(default_factory=SeparatorDefaults)
    voice_converters: VoiceConverterDefaults = field(default_factory=VoiceConverterDefaults)
    effects: EffectsDefaults = field(default_factory=EffectsDefaults)
    converters: ConverterDefaults = field(default_factory=ConverterDefaults)
    audio_utils: AudioUtilsDefaults = field(default_factory=AudioUtilsDefaults)
    security: SecurityDefaults = field(default_factory=SecurityDefaults)
    logging: LoggingDefaults = field(default_factory=LoggingDefaults)

    def to_dict(self) -> dict:
        """转换为字典用于 JSON 序列化"""
        return {k: v.to_dict() if hasattr(v, 'to_dict') else v
                for k, v in self.__dict__.items()
                if not k.startswith('_')}


# 默认配置实例
DEFAULT_CONFIG = SomaDefaults()
