"""
RVC Training Configuration Module

Provides hierarchical configuration for RVC model training:
- DataConfig: Audio data preprocessing parameters
- ModelConfig: Network architecture parameters
- OptimizerConfig: Optimizer and learning rate parameters
- TrainConfig: Training loop parameters
- F0Config: F0 extraction parameters
- TrainingConfig: Top-level configuration aggregating all sub-configs
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Audio data preprocessing configuration."""

    sample_rate: int = 40000
    hop_length: int = 512
    n_mels: int = 128
    n_fft: int = 2048
    win_length: int = 2048
    segment_duration: float = 3.0  # seconds per training segment
    overlap: float = 0.1  # overlap ratio between segments
    max_clip_duration: float = 30.0  # max duration for a single clip
    min_clip_duration: float = 0.5  # min duration for a single clip
    silence_threshold: float = -40.0  # dB, threshold for silence detection
    normalize_mode: str = "peak"  # "peak" or "lufs"
    target_lufs: float = -23.0  # target LUFS for normalization
    target_peak: float = 0.95  # target peak amplitude for normalization
    supported_formats: List[str] = field(
        default_factory=lambda: ["wav", "mp3", "flac", "ogg", "m4a"]
    )

    def validate(self) -> List[str]:
        """Validate configuration parameters. Returns list of error messages."""
        errors = []
        if self.sample_rate <= 0:
            errors.append(f"sample_rate must be positive, got {self.sample_rate}")
        if self.hop_length <= 0:
            errors.append(f"hop_length must be positive, got {self.hop_length}")
        if self.n_mels <= 0:
            errors.append(f"n_mels must be positive, got {self.n_mels}")
        if self.n_fft <= 0:
            errors.append(f"n_fft must be positive, got {self.n_fft}")
        if self.segment_duration <= 0:
            errors.append(
                f"segment_duration must be positive, got {self.segment_duration}"
            )
        if not 0 <= self.overlap < 1:
            errors.append(f"overlap must be in [0, 1), got {self.overlap}")
        if self.normalize_mode not in ("peak", "lufs"):
            errors.append(
                f"normalize_mode must be 'peak' or 'lufs', got {self.normalize_mode}"
            )
        if self.target_peak <= 0 or self.target_peak > 1.0:
            errors.append(
                f"target_peak must be in (0, 1], got {self.target_peak}"
            )
        return errors


@dataclass
class ModelConfig:
    """RVC network architecture configuration."""

    in_channels: int = 256
    out_channels: int = 1
    hidden_channels: int = 256
    kernel_size: int = 7
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    embed_dim: int = 256
    pitch_encoder_dim: int = 256
    use_flow: bool = False
    # Discriminator config
    mpd_periods: List[List[int]] = field(
        default_factory=lambda: [[2, 3, 5, 7], [1, 5, 7, 11], [1, 7, 13, 19]]
    )

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        if self.in_channels <= 0:
            errors.append(f"in_channels must be positive, got {self.in_channels}")
        if self.hidden_channels <= 0:
            errors.append(
                f"hidden_channels must be positive, got {self.hidden_channels}"
            )
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            errors.append(
                "upsample_rates and upsample_kernel_sizes must have same length"
            )
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            errors.append(
                "resblock_kernel_sizes and resblock_dilation_sizes must have same length"
            )
        return errors


@dataclass
class OptimizerConfig:
    """Optimizer and learning rate configuration."""

    optimizer: str = "adamw"  # "adam" or "adamw"
    lr: float = 2e-4
    betas: List[float] = field(default_factory=lambda: [0.8, 0.99])
    weight_decay: float = 0.01
    eps: float = 1e-9
    # LR scheduler
    scheduler: str = "cosine"  # "cosine" or "step"
    warmup_steps: int = 1000
    total_steps: int = 500000
    min_lr: float = 1e-6
    # Gradient
    grad_clip: float = 1.0

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        if self.lr <= 0:
            errors.append(f"lr must be positive, got {self.lr}")
        if self.optimizer not in ("adam", "adamw"):
            errors.append(f"optimizer must be 'adam' or 'adamw', got {self.optimizer}")
        if self.scheduler not in ("cosine", "step"):
            errors.append(f"scheduler must be 'cosine' or 'step', got {self.scheduler}")
        if self.grad_clip <= 0:
            errors.append(f"grad_clip must be positive, got {self.grad_clip}")
        if self.warmup_steps < 0:
            errors.append(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        return errors


@dataclass
class TrainConfig:
    """Training loop configuration."""

    batch_size: int = 8
    num_epochs: int = 1000
    save_interval: int = 5  # save checkpoint every N epochs
    log_interval: int = 100  # log every N steps
    eval_interval: int = 5  # evaluate every N epochs
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    # AMP
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    # Paths
    output_dir: str = "output/rvc_training"
    log_dir: str = "output/rvc_training/logs"
    checkpoint_dir: str = "output/rvc_training/checkpoints"
    # Loss weights
    mel_loss_weight: float = 45.0
    l1_loss_weight: float = 1.0
    adv_loss_weight: float = 1.0
    fm_loss_weight: float = 2.0
    # Multi-GPU
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            errors.append(f"num_epochs must be positive, got {self.num_epochs}")
        if self.num_workers < 0:
            errors.append(
                f"num_workers must be non-negative, got {self.num_workers}"
            )
        if self.use_amp and self.amp_dtype not in ("float16", "bfloat16"):
            errors.append(
                f"amp_dtype must be 'float16' or 'bfloat16', got {self.amp_dtype}"
            )
        if self.mel_loss_weight < 0:
            errors.append(
                f"mel_loss_weight must be non-negative, got {self.mel_loss_weight}"
            )
        return errors


@dataclass
class F0Config:
    """F0 extraction configuration."""

    method: str = "dio"  # "pm", "dio", "harvest", "crepe"
    f0_min: float = 50.0
    f0_max: float = 1100.0
    voiced_threshold: float = 0.5
    use_uv: bool = True  # use unvoiced/voiced flag

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        valid_methods = ("pm", "dio", "harvest", "crepe")
        if self.method not in valid_methods:
            errors.append(
                f"method must be one of {valid_methods}, got {self.method}"
            )
        if self.f0_min <= 0:
            errors.append(f"f0_min must be positive, got {self.f0_min}")
        if self.f0_max <= self.f0_min:
            errors.append(
                f"f0_max must be > f0_min, got f0_max={self.f0_max}, f0_min={self.f0_min}"
            )
        return errors


@dataclass
class TrainingConfig:
    """
    Top-level training configuration.

    Aggregates all sub-configs and provides JSON import/export.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    f0: F0Config = field(default_factory=F0Config)

    def validate(self) -> List[str]:
        """Validate all sub-configs. Returns list of error messages."""
        errors = []
        errors.extend(self.data.validate())
        errors.extend(self.model.validate())
        errors.extend(self.optimizer.validate())
        errors.extend(self.train.validate())
        errors.extend(self.f0.validate())
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        config = cls()
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "optimizer" in data:
            config.optimizer = OptimizerConfig(**data["optimizer"])
        if "train" in data:
            config.train = TrainConfig(**data["train"])
        if "f0" in data:
            config.f0 = F0Config(**data["f0"])
        return config

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Configuration saved to %s", path)

    @classmethod
    def load_json(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = cls.from_dict(data)
        errors = config.validate()
        if errors:
            logger.warning("Configuration validation warnings: %s", errors)
        logger.info("Configuration loaded from %s", path)
        return config

    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = self.validate()
        if errors:
            logger.warning("TrainingConfig validation warnings: %s", errors)
