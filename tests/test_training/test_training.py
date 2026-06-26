"""
Tests for RVC Training Module

Covers:
- config: TrainingConfig, DataConfig, ModelConfig, OptimizerConfig, TrainConfig, F0Config
- preprocess: AudioPreprocessor, load_audio, trim_silence, normalize, segment, validate
- dataset: RVCDataset, split_dataset
- trainer: RVCTrainer, MultiPeriodDiscriminator, loss functions
- cli: Argument parser, subcommands
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ============================================================================
# Config Tests
# ============================================================================

class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        from src.training.config import DataConfig
        config = DataConfig()
        assert config.sample_rate == 40000
        assert config.hop_length == 512
        assert config.n_mels == 128
        assert config.segment_duration == 3.0
        assert config.overlap == 0.1
        assert config.normalize_mode == "peak"

    def test_custom_values(self):
        from src.training.config import DataConfig
        config = DataConfig(sample_rate=48000, hop_length=256, n_mels=80)
        assert config.sample_rate == 48000
        assert config.hop_length == 256
        assert config.n_mels == 80

    def test_validate_valid(self):
        from src.training.config import DataConfig
        config = DataConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_sample_rate(self):
        from src.training.config import DataConfig
        config = DataConfig(sample_rate=-1)
        errors = config.validate()
        assert len(errors) > 0
        assert any("sample_rate" in e for e in errors)

    def test_validate_invalid_overlap(self):
        from src.training.config import DataConfig
        config = DataConfig(overlap=1.5)
        errors = config.validate()
        assert len(errors) > 0
        assert any("overlap" in e for e in errors)

    def test_validate_invalid_normalize_mode(self):
        from src.training.config import DataConfig
        config = DataConfig(normalize_mode="invalid")
        errors = config.validate()
        assert len(errors) > 0


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        from src.training.config import ModelConfig
        config = ModelConfig()
        assert config.in_channels == 256
        assert config.hidden_channels == 256
        assert config.use_flow is False

    def test_validate_valid(self):
        from src.training.config import ModelConfig
        config = ModelConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_mismatched_lengths(self):
        from src.training.config import ModelConfig
        config = ModelConfig(
            upsample_rates=[8, 8],
            upsample_kernel_sizes=[16],
        )
        errors = config.validate()
        assert len(errors) > 0


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""

    def test_default_values(self):
        from src.training.config import OptimizerConfig
        config = OptimizerConfig()
        assert config.optimizer == "adamw"
        assert config.lr == 2e-4
        assert config.scheduler == "cosine"

    def test_validate_invalid_optimizer(self):
        from src.training.config import OptimizerConfig
        config = OptimizerConfig(optimizer="sgd")
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_invalid_lr(self):
        from src.training.config import OptimizerConfig
        config = OptimizerConfig(lr=-0.001)
        errors = config.validate()
        assert len(errors) > 0


class TestTrainConfig:
    """Tests for TrainConfig."""

    def test_default_values(self):
        from src.training.config import TrainConfig
        config = TrainConfig()
        assert config.batch_size == 8
        assert config.num_epochs == 1000
        assert config.use_amp is True

    def test_validate_invalid_batch_size(self):
        from src.training.config import TrainConfig
        config = TrainConfig(batch_size=0)
        errors = config.validate()
        assert len(errors) > 0


class TestF0Config:
    """Tests for F0Config."""

    def test_default_values(self):
        from src.training.config import F0Config
        config = F0Config()
        assert config.method == "dio"
        assert config.f0_min == 50.0
        assert config.f0_max == 1100.0

    def test_validate_invalid_method(self):
        from src.training.config import F0Config
        config = F0Config(method="invalid")
        errors = config.validate()
        assert len(errors) > 0

    def test_validate_f0_range(self):
        from src.training.config import F0Config
        config = F0Config(f0_min=1000, f0_max=500)
        errors = config.validate()
        assert len(errors) > 0


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_creation(self):
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        assert config.data.sample_rate == 40000
        assert config.model.in_channels == 256
        assert config.optimizer.lr == 2e-4
        assert config.train.batch_size == 8
        assert config.f0.method == "dio"

    def test_validate_all_valid(self):
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_to_dict(self):
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "data" in d
        assert "model" in d
        assert "optimizer" in d
        assert "train" in d
        assert "f0" in d
        assert d["data"]["sample_rate"] == 40000

    def test_from_dict(self):
        from src.training.config import TrainingConfig
        data = {
            "data": {"sample_rate": 48000, "hop_length": 256},
            "model": {"in_channels": 128},
            "optimizer": {"lr": 1e-4},
            "train": {"batch_size": 16},
            "f0": {"method": "pm"},
        }
        config = TrainingConfig.from_dict(data)
        assert config.data.sample_rate == 48000
        assert config.data.hop_length == 256
        assert config.model.in_channels == 128
        assert config.optimizer.lr == 1e-4
        assert config.train.batch_size == 16
        assert config.f0.method == "pm"

    def test_save_and_load_json(self):
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        config.data.sample_rate = 48000
        config.train.batch_size = 16

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config.save_json(path)

            loaded = TrainingConfig.load_json(path)
            assert loaded.data.sample_rate == 48000
            assert loaded.train.batch_size == 16

    def test_load_nonexistent_json(self):
        from src.training.config import TrainingConfig
        with pytest.raises(FileNotFoundError):
            TrainingConfig.load_json("/nonexistent/path/config.json")


# ============================================================================
# Preprocess Tests
# ============================================================================

class TestPreprocessFunctions:
    """Tests for preprocessing functions."""

    def test_compute_rms(self):
        from src.training.preprocess import compute_rms
        audio = np.ones(1000, dtype=np.float32) * 0.5
        rms = compute_rms(audio)
        assert abs(rms - 0.5) < 1e-5

    def test_compute_rms_empty(self):
        from src.training.preprocess import compute_rms
        rms = compute_rms(np.array([], dtype=np.float32))
        assert rms == 0.0

    def test_compute_snr(self):
        from src.training.preprocess import compute_snr
        # Create signal with clear voiced and silent regions
        np.random.seed(42)
        # Voiced region: strong sine wave
        voiced = np.sin(2 * np.pi * 440 * np.arange(20000) / 40000).astype(np.float32) * 0.5
        # Silent region: very quiet noise
        silent = np.random.randn(20000).astype(np.float32) * 0.001
        audio = np.concatenate([silent, voiced])
        snr = compute_snr(audio)
        assert snr > 5.0  # Should have reasonable SNR

    def test_trim_silence(self):
        from src.training.preprocess import trim_silence
        # Create audio with silence at beginning and end
        # Need enough silence for the algorithm to detect (min_silence_duration=0.3s)
        silence = np.zeros(16000, dtype=np.float32)  # 0.4s at 40kHz
        signal = np.random.randn(32000).astype(np.float32) * 0.5
        audio = np.concatenate([silence, signal, silence])
        trimmed = trim_silence(audio, 40000, threshold_db=-40.0, min_silence_duration=0.1)
        assert len(trimmed) < len(audio)

    def test_trim_silence_empty(self):
        from src.training.preprocess import trim_silence
        audio = np.array([], dtype=np.float32)
        trimmed = trim_silence(audio, 40000)
        assert len(trimmed) == 0

    def test_normalize_peak(self):
        from src.training.preprocess import normalize_peak
        audio = np.array([0.1, -0.5, 0.3, -0.2], dtype=np.float32)
        normalized = normalize_peak(audio, target_peak=0.95)
        assert abs(np.max(np.abs(normalized)) - 0.95) < 1e-5

    def test_normalize_peak_silent(self):
        from src.training.preprocess import normalize_peak
        audio = np.zeros(100, dtype=np.float32)
        normalized = normalize_peak(audio, target_peak=0.95)
        assert np.allclose(normalized, 0.0)

    def test_normalize_lufs(self):
        from src.training.preprocess import normalize_lufs
        audio = np.random.randn(40000).astype(np.float32) * 0.1
        normalized = normalize_lufs(audio, 40000, target_lufs=-23.0)
        assert len(normalized) == len(audio)
        # Should not clip
        assert np.max(np.abs(normalized)) <= 1.0

    def test_segment_audio(self):
        from src.training.preprocess import segment_audio
        audio = np.random.randn(120000).astype(np.float32)  # 3 seconds at 40kHz
        segments = segment_audio(audio, 40000, segment_duration=1.0, overlap=0.0)
        assert len(segments) == 3
        assert all(len(s) == 40000 for s in segments)

    def test_segment_audio_with_overlap(self):
        from src.training.preprocess import segment_audio
        audio = np.random.randn(160000).astype(np.float32)  # 4 seconds at 40kHz
        segments = segment_audio(audio, 40000, segment_duration=2.0, overlap=0.5)
        assert len(segments) >= 2
        assert all(len(s) == 80000 for s in segments)

    def test_segment_audio_empty(self):
        from src.training.preprocess import segment_audio
        segments = segment_audio(np.array([], dtype=np.float32), 40000)
        assert len(segments) == 0

    def test_segment_audio_short(self):
        from src.training.preprocess import segment_audio
        audio = np.random.randn(1000).astype(np.float32)  # Very short
        segments = segment_audio(audio, 40000, segment_duration=3.0)
        assert len(segments) == 0  # Too short for even one segment

    def test_validate_audio_quality_valid(self):
        from src.training.preprocess import validate_audio_quality
        # Use a controlled signal that passes all quality checks
        audio = np.sin(2 * np.pi * 440 * np.arange(40000) / 40000).astype(np.float32) * 0.5
        is_valid, issues = validate_audio_quality(audio, 40000, min_snr=0.0)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_audio_quality_too_short(self):
        from src.training.preprocess import validate_audio_quality
        audio = np.random.randn(100).astype(np.float32)
        is_valid, issues = validate_audio_quality(audio, 40000, min_duration=0.5)
        assert is_valid is False
        assert any("short" in i.lower() for i in issues)

    def test_validate_audio_quality_silent(self):
        from src.training.preprocess import validate_audio_quality
        audio = np.zeros(40000, dtype=np.float32)
        is_valid, issues = validate_audio_quality(audio, 40000)
        assert is_valid is False
        assert any("silent" in i.lower() for i in issues)

    def test_validate_audio_quality_empty(self):
        from src.training.preprocess import validate_audio_quality
        audio = np.array([], dtype=np.float32)
        is_valid, issues = validate_audio_quality(audio, 40000)
        assert is_valid is False
        assert any("empty" in i.lower() for i in issues)

    def test_load_audio_nonexistent(self):
        from src.training.preprocess import load_audio
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/audio.wav")

    def test_load_audio_unsupported_format(self):
        from src.training.preprocess import load_audio
        with tempfile.NamedTemporaryFile(suffix=".xyz") as f:
            with pytest.raises(ValueError, match="Unsupported"):
                load_audio(f.name)


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor class."""

    def test_creation_default(self):
        from src.training.preprocess import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        assert preprocessor.config.sample_rate == 40000

    def test_creation_custom_config(self):
        from src.training.config import DataConfig
        from src.training.preprocess import AudioPreprocessor
        config = DataConfig(sample_rate=48000)
        preprocessor = AudioPreprocessor(config)
        assert preprocessor.config.sample_rate == 48000


# ============================================================================
# Dataset Tests
# ============================================================================

class TestRVCDataset:
    """Tests for RVCDataset."""

    def test_creation_empty_dir(self):
        from src.training.dataset import RVCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = RVCDataset(tmpdir)
            assert len(dataset) == 0

    def test_creation_with_data(self):
        from src.training.dataset import RVCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy .npy files
            for i in range(5):
                data = np.random.randn(120000).astype(np.float32)
                np.save(os.path.join(tmpdir, f"seg_{i:04d}.npy"), data)

            dataset = RVCDataset(tmpdir)
            assert len(dataset) == 5

    def test_getitem(self):
        from src.training.dataset import RVCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.randn(120000).astype(np.float32)
            np.save(os.path.join(tmpdir, "seg_0000.npy"), data)

            dataset = RVCDataset(tmpdir)
            sample = dataset[0]
            assert "audio" in sample
            assert "mel" in sample
            assert isinstance(sample["audio"], np.ndarray)
            assert isinstance(sample["mel"], np.ndarray)

    def test_getitem_padding(self):
        from src.training.dataset import RVCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            # Short audio that needs padding
            data = np.random.randn(1000).astype(np.float32)
            np.save(os.path.join(tmpdir, "seg_0000.npy"), data)

            dataset = RVCDataset(tmpdir, segment_length=120000)
            sample = dataset[0]
            assert len(sample["audio"]) == 120000

    def test_getitem_truncation(self):
        from src.training.dataset import RVCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            # Long audio that needs truncation
            data = np.random.randn(500000).astype(np.float32)
            np.save(os.path.join(tmpdir, "seg_0000.npy"), data)

            dataset = RVCDataset(tmpdir, segment_length=120000)
            sample = dataset[0]
            assert len(sample["audio"]) == 120000

    def test_getitem_out_of_range(self):
        from src.training.dataset import RVCDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = RVCDataset(tmpdir)
            with pytest.raises(IndexError):
                dataset[0]


class TestSplitDataset:
    """Tests for split_dataset."""

    def test_split_empty(self):
        from src.training.dataset import split_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            train, val = split_dataset(tmpdir)
            assert len(train) == 0
            assert len(val) == 0

    def test_split_ratio(self):
        from src.training.dataset import split_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                np.save(os.path.join(tmpdir, f"seg_{i:04d}.npy"), np.zeros(100))

            train, val = split_dataset(tmpdir, train_ratio=0.8, seed=42)
            assert len(train) == 8
            assert len(val) == 2

    def test_split_reproducible(self):
        from src.training.dataset import split_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                np.save(os.path.join(tmpdir, f"seg_{i:04d}.npy"), np.zeros(100))

            train1, val1 = split_dataset(tmpdir, seed=42)
            train2, val2 = split_dataset(tmpdir, seed=42)
            assert train1 == train2
            assert val1 == val2


# ============================================================================
# Trainer Tests
# ============================================================================

class TestLossFunctions:
    """Tests for loss functions."""

    def test_mel_spectrogram_loss(self):
        import torch
        from src.training.trainer import mel_spectrogram_loss
        y_pred = torch.randn(2, 16000)
        y_true = torch.randn(2, 16000)
        loss = mel_spectrogram_loss(y_pred, y_true)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_mel_spectrogram_loss_identical(self):
        import torch
        from src.training.trainer import mel_spectrogram_loss
        y = torch.randn(2, 16000)
        loss = mel_spectrogram_loss(y, y)
        assert loss.item() < 1e-5

    def test_feature_matching_loss(self):
        import torch
        from src.training.trainer import feature_matching_loss
        fake_feats = [[torch.randn(2, 32, 10)], [torch.randn(2, 64, 10)]]
        real_feats = [[torch.randn(2, 32, 10)], [torch.randn(2, 64, 10)]]
        loss = feature_matching_loss(fake_feats, real_feats)
        assert loss >= 0

    def test_discriminator_loss(self):
        import torch
        from src.training.trainer import discriminator_loss
        real_out = [torch.randn(2, 1, 10)]
        fake_out = [torch.randn(2, 1, 10)]
        loss = discriminator_loss(real_out, fake_out)
        assert loss >= 0

    def test_generator_adversarial_loss(self):
        import torch
        from src.training.trainer import generator_adversarial_loss
        fake_out = [torch.randn(2, 1, 10)]
        loss = generator_adversarial_loss(fake_out)
        assert isinstance(loss.item(), float)


class TestRVCTrainer:
    """Tests for RVCTrainer."""

    def test_creation(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer
        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        assert trainer.device == "cpu"
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_build_models(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer
        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()
        assert trainer.generator is not None
        assert trainer.discriminator is not None

    def test_build_optimizers(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer
        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()
        trainer.build_optimizers()
        assert trainer.optimizer_g is not None
        assert trainer.optimizer_d is not None
        assert trainer.scheduler_g is not None
        assert trainer.scheduler_d is not None

    def test_save_and_load_checkpoint(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig()
            config.train.checkpoint_dir = tmpdir

            trainer = RVCTrainer(config, device="cpu")
            trainer.build_models()
            trainer.build_optimizers()
            trainer.global_step = 100

            # Save
            trainer.save_checkpoint(5, {"g_loss": 1.0})
            ckpt_path = os.path.join(tmpdir, "checkpoint_latest.pt")

            # Load into new trainer
            trainer2 = RVCTrainer(config, device="cpu")
            trainer2.build_models()
            trainer2.build_optimizers()
            epoch = trainer2.load_checkpoint(ckpt_path)
            assert epoch == 5
            assert trainer2.global_step == 100

    def test_export_for_inference(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "exported.pt")
            trainer.export_for_inference(output_path)
            assert os.path.exists(output_path)

            # Verify checkpoint format
            import torch
            ckpt = torch.load(output_path, map_location="cpu")
            assert "model" in ckpt
            assert "config" in ckpt
            assert ckpt["config"]["sample_rate"] == 40000

    def test_export_without_model_raises(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        with pytest.raises(RuntimeError, match="Generator not built"):
            trainer.export_for_inference("output.pt")


# ============================================================================
# CLI Tests
# ============================================================================

class TestCLI:
    """Tests for CLI module."""

    def test_build_parser(self):
        from src.training.cli import build_parser
        parser = build_parser()
        assert parser is not None

    def test_parse_preprocess(self):
        from src.training.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["preprocess", "input/", "output/"])
        assert args.command == "preprocess"
        assert args.input_dir == "input/"
        assert args.output_dir == "output/"

    def test_parse_preprocess_with_options(self):
        from src.training.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "preprocess", "input/", "output/",
            "--sample-rate", "48000",
            "--segment-duration", "5.0",
            "--overlap", "0.2",
        ])
        assert args.sample_rate == 48000
        assert args.segment_duration == 5.0
        assert args.overlap == 0.2

    def test_parse_train(self):
        from src.training.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["train", "data/"])
        assert args.command == "train"
        assert args.data_dir == "data/"

    def test_parse_train_with_options(self):
        from src.training.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "train", "data/",
            "--batch-size", "16",
            "--epochs", "50",
            "--device", "cpu",
        ])
        assert args.batch_size == 16
        assert args.epochs == 50
        assert args.device == "cpu"

    def test_parse_export(self):
        from src.training.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "export", "--checkpoint", "model.pt",
        ])
        assert args.command == "export"
        assert args.checkpoint == "model.pt"

    def test_main_no_command(self):
        from src.training.cli import main
        # Should not raise, just print help
        main([])

    def test_main_verbose(self):
        from src.training.cli import main
        main(["-v"])


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the training pipeline."""

    def test_full_config_roundtrip(self):
        """Test saving and loading a full configuration."""
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        config.data.sample_rate = 48000
        config.model.hidden_channels = 192
        config.optimizer.lr = 1e-4
        config.train.batch_size = 4
        config.f0.method = "pm"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config.save_json(path)
            loaded = TrainingConfig.load_json(path)

            assert loaded.data.sample_rate == 48000
            assert loaded.model.hidden_channels == 192
            assert loaded.optimizer.lr == 1e-4
            assert loaded.train.batch_size == 4
            assert loaded.f0.method == "pm"

    def test_preprocess_and_dataset(self):
        """Test preprocessing followed by dataset loading."""
        from src.training.config import DataConfig
        from src.training.dataset import RVCDataset
        from src.training.preprocess import AudioPreprocessor, segment_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy audio segments
            audio = np.random.randn(120000).astype(np.float32) * 0.5
            segments = segment_audio(audio, 40000, segment_duration=1.0, overlap=0.0)

            for i, seg in enumerate(segments):
                np.save(os.path.join(tmpdir, f"seg_{i:04d}.npy"), seg)

            # Load as dataset with explicit segment_length
            dataset = RVCDataset(tmpdir, segment_length=40000)
            assert len(dataset) == 3

            sample = dataset[0]
            assert sample["audio"].shape[0] == 40000  # 1 second at 40kHz

    def test_trainer_export_compatibility(self):
        """Test that exported model has correct format for RVCConverter."""
        import torch
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")
            trainer.export_for_inference(output_path)

            # Verify format matches RVCConverter expectations
            ckpt = torch.load(output_path, map_location="cpu")
            assert "model" in ckpt or "weight" in ckpt
            assert "config" in ckpt
            assert ckpt["config"]["sample_rate"] == config.data.sample_rate
            assert ckpt["config"]["hop_length"] == config.data.hop_length
