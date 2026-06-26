"""
RVC Training Module

Provides a complete training pipeline for RVC (Retrieval-Based Voice Conversion)
dry voice models within the SOMA Audio AI framework.

Modules:
- config: Hierarchical training configuration
- preprocess: Audio preprocessing pipeline
- dataset: PyTorch-compatible dataset and dataloader
- trainer: Training loop with AMP, checkpointing, and TensorBoard
- cli: Command-line interface

Example:
    from training import TrainingConfig, AudioPreprocessor, RVCTrainer

    # Configure
    config = TrainingConfig()
    config.data.sample_rate = 40000

    # Preprocess
    preprocessor = AudioPreprocessor(config.data)
    preprocessor.process_directory("raw_audio/", "processed/")

    # Train
    trainer = RVCTrainer(config)
    trainer.build_models()
    trainer.build_optimizers()
    trainer.train(dataloader)

    # Export
    trainer.export_for_inference("output/model.pt")
"""

from .config import (
    DataConfig,
    F0Config,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
    TrainingConfig,
)
from .preprocess import (
    AudioPreprocessor,
    load_audio,
    normalize_lufs,
    normalize_peak,
    segment_audio,
    trim_silence,
    validate_audio_quality,
)
from .dataset import RVCDataset, create_dataloader, split_dataset
from .trainer import RVCTrainer
from .feature_extractor import HuBERTFeatureExtractor, F0Extractor, FeaturePipeline
from .inference import RVCInference, VocoderWrapper
from .cli import main as cli_main

__all__ = [
    # Config
    "DataConfig",
    "F0Config",
    "ModelConfig",
    "OptimizerConfig",
    "TrainConfig",
    "TrainingConfig",
    # Preprocess
    "AudioPreprocessor",
    "load_audio",
    "normalize_lufs",
    "normalize_peak",
    "segment_audio",
    "trim_silence",
    "validate_audio_quality",
    # Dataset
    "RVCDataset",
    "create_dataloader",
    "split_dataset",
    # Trainer
    "RVCTrainer",
    # Feature Extraction
    "HuBERTFeatureExtractor",
    "F0Extractor",
    "FeaturePipeline",
    # Inference
    "RVCInference",
    "VocoderWrapper",
    # CLI
    "cli_main",
]
