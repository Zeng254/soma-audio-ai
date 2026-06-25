"""
SOMA Configuration center - Default configuration parameters

This module centrally manages default configuration parameters for all modules, including:
- Separator (Separators)
- Voice converter (Voice Converters)
- Audio effects processor (Effects)
- Format converter (Converters)
- Audio tools (Audio Utils)
- Security settings (Security)
- Logging settings (Logging)

All hardcoded default values should be migrated to this module.
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class SeparatorDefaults:
    """Separator default configuration"""
    # Model related
    default_model: str = "htdemucs_ft"
    models_cache_dir: str = "~/.cache/torch/hub/demucs"
    device: str = "auto"  # auto, cpu, cuda, mps

    # Processing parameters
    segment_size: int = 0  # 0 means no segmentation, -1 means automatic
    overlap: float = 0.5  # Overlap rate 0-1
    batch_size: int = 1
    num_workers: int = 0  # 0 means use main process

    # Output related
    output_format: str = "wav"  # wav, mp3, flac, ogg
    output_bit_depth: int = 32  # 16, 24, 32
    output_sample_rate: int = 44100

    # Specific model parameters
    stem_types: list = field(default_factory=lambda: ["vocals", "drums", "bass", "other"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


@dataclass
class VoiceConverterDefaults:
    """Voice converter default configuration"""
    # Engine related
    default_engine: str = "auto"  # auto, rvc, sovits
    device: str = "auto"

    # General processing parameters
    pitch_shift: float = 0.0  # Semitones
    pitch_algo: str = "rmvpe"  # rmvpe, crepe, dio, harvest, pm
    vpm: float = 0.5  # Voice timbre match 0-1
    rms_mix: float = 0.5  # Loudness mix 0-1

    # Quality parameters
    f0_smooth: int = 0  # Pitch smoothing frames
    protect: float = 0.33  # Protect non-speech parts

    # SoVITS specific parameters
    sovits_version: str = "4.1"
    diffusion_steps: int = 10  # Diffusion steps
    diffusion_seed: int = 42

    # RVC specific parameters
    rvc_version: str = "v2"
    index_rate: float = 0.3  # Index search ratio
    filter_radius: int = 3  # Filter radius

    # Audio parameters
    input_sample_rate: int = 44100
    output_sample_rate: int = 44100
    hop_length: int = 128

    # Model cache
    models_cache_dir: str = "~/.soma/models"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
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
    """Audio effects default configuration"""

    @dataclass
    class EQDefaults:
        """Equalizer default configuration"""
        enabled: bool = True
        sample_rate: int = 44100
        bands: int = 10  # Number of bands

        # Band configuration (10 bands)
        band_frequencies: list = field(default_factory=lambda: [
            31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000
        ])

        # Default gain (dB)
        default_gains: list = field(default_factory=lambda: [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])

        # Preset type
        preset: str = "flat"  # flat, vocal_enhance, bass_boost, treble_boost

        def to_dict(self) -> dict:
            """Convert to dictionary for JSON serialization"""
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith('_')}

    @dataclass
    class ReverbDefaults:
        """Reverb effect default configuration"""
        enabled: bool = True
        sample_rate: int = 44100

        # Reverb type
        reverb_type: str = "room"  # room, hall, plate, cathedral

        # Room parameters
        room_size: float = 0.5  # 0-1
        damping: float = 0.5  # 0-1
        wet_level: float = 0.3  # 0-1
        dry_level: float = 0.7  # 0-1
        width: float = 1.0  # 0-1
        freeze_mode: float = 0.0  # 0-1

        # Preset
        preset: str = "natural"  # natural, dramatic, subtle, wet

        def to_dict(self) -> dict:
            """Convert to dictionary for JSON serialization"""
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith('_')}

    @dataclass
    class PitchDefaults:
        """Pitch shifting default configuration"""
        enabled: bool = True
        sample_rate: int = 44100

        # Pitch parameters
        semitones: float = 0.0  # Semitones
        cents: float = 0.0  # Cents adjustment
        formants_shift: float = 0.0  # Formant shift

        # Algorithm parameters
        algorithm: str = "soundtouch"  # soundtouch, librosa, pitchy
        quick: bool = False  # Quick mode
        fft_size: int = 2048

        def to_dict(self) -> dict:
            """Convert to dictionary for JSON serialization"""
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith('_')}

    eq: EQDefaults = field(default_factory=EQDefaults)
    reverb: ReverbDefaults = field(default_factory=ReverbDefaults)
    pitch: PitchDefaults = field(default_factory=PitchDefaults)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
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
    """Format converter default configuration"""
    # Input parameters
    input_format: str = "auto"  # Auto detect format

    # Output parameters
    output_format: str = "wav"
    output_bit_depth: int = 16  # 8, 16, 24, 32
    output_sample_rate: int = 44100
    output_channels: int = 2  # 1 Mono, 2 Stereo

    # Encoding parameters
    codec: str = "pcm_s16le"  # libmp3lame, aac, opus, pcm_s16le
    bitrate: str = "192k"  # 128k, 192k, 320k
    quality: int = 5  # 0-9 (VBR), 0 is best

    # ffmpeg parameters
    ffmpeg_path: str = "ffmpeg"
    timeout: int = 300  # seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class AudioUtilsDefaults:
    """Audio tools default configuration"""
    # Audio parameters
    default_sample_rate: int = 44100
    default_channels: int = 2
    default_bit_depth: int = 16

    # File size limits
    min_file_size_bytes: int = 1024  # Minimum file size 1KB
    max_file_size_mb: int = 500  # 500MB
    max_duration_seconds: int = 3600  # 1 hour

    # Cache settings
    cache_enabled: bool = True
    cache_dir: str = "~/.soma/cache"
    cache_max_size_mb: int = 1024  # 1GB

    # Temporary files
    temp_dir: str = "~/.soma/temp"
    auto_cleanup: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class SecurityDefaults:
    """Security settings default configuration"""
    # Path restrictions
    allowed_base_dirs: list = field(default_factory=lambda: [
        "~/.soma/workspace",
        "~/Documents",
        "~/Music",
        "/tmp"
    ])
    allow_symlinks: bool = False
    max_path_depth: int = 20

    # File validation
    allowed_audio_formats: list = field(default_factory=lambda: [
        "wav", "mp3", "flac", "ogg", "m4a", "aac", "wma"
    ])
    max_file_size_mb: int = 500
    min_file_size_bytes: int = 1024  # 1KB

    # Model validation
    max_model_size_mb: int = 5000  # 5GB
    trusted_model_signatures: list = field(default_factory=list)

    # Network security
    allow_network_access: bool = True  # Whether to allow network access
    allowed_hosts: list = field(default_factory=lambda: [
        "huggingface.co",
        "github.com"
    ])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class LoggingDefaults:
    """Logging system default configuration"""
    # Logging level
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Logging output
    console_output: bool = True
    file_output: bool = True

    # Logging file
    log_dir: str = "~/.soma/logs"
    log_file_name: str = "soma.log"
    max_log_size_mb: int = 10
    backup_count: int = 7  # Keep 7 days

    # Logging format
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Module-level logging
    module_levels: Dict[str, str] = field(default_factory=lambda: {
        "soma": "INFO",
        "soma.separators": "INFO",
        "soma.voice_converters": "INFO",
        "soma.effects": "INFO",
        "soma.pipeline": "DEBUG"
    })

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class SomaDefaults:
    """SOMA main configuration"""
    # Version info
    app_name: str = "SOMA"
    version: str = "0.1.0"

    # Sub-configurations (plural form, consistent with Config API)
    separators: SeparatorDefaults = field(default_factory=SeparatorDefaults)
    voice_converters: VoiceConverterDefaults = field(default_factory=VoiceConverterDefaults)
    effects: EffectsDefaults = field(default_factory=EffectsDefaults)
    converters: ConverterDefaults = field(default_factory=ConverterDefaults)
    audio_utils: AudioUtilsDefaults = field(default_factory=AudioUtilsDefaults)
    security: SecurityDefaults = field(default_factory=SecurityDefaults)
    logging: LoggingDefaults = field(default_factory=LoggingDefaults)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {k: v.to_dict() if hasattr(v, 'to_dict') else v
                for k, v in self.__dict__.items()
                if not k.startswith('_')}


# Default configuration instance
DEFAULT_CONFIG = SomaDefaults()
