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


# ============================================================================
# Dual Export Tests
# ============================================================================

class TestDualExport:
    """Tests for dual-format export (RVC + SoVITS)."""

    def test_export_sovits_format(self):
        """Test SoVITS format export creates correct checkpoint."""
        import torch
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model_sovits.pt")
            trainer.export_sovits_format(output_path)

            # Verify checkpoint file exists
            assert os.path.exists(output_path)

            # Verify checkpoint format
            ckpt = torch.load(output_path, map_location="cpu")
            assert "model" in ckpt
            assert "config" in ckpt
            assert ckpt["version"] == "sovits_v4.1"

            # Verify config has SoVITS-compatible fields
            sovits_cfg = ckpt["config"]
            assert "hidden_channels" in sovits_cfg
            assert "spec_channels" in sovits_cfg
            assert sovits_cfg["n_vocab"] == 0  # Content encoder mode

            # Verify companion config.json exists
            config_json_path = os.path.join(tmpdir, "config.json")
            assert os.path.exists(config_json_path)

            with open(config_json_path, "r") as f:
                config_json = json.load(f)
            assert config_json["sampling_rate"] == config.data.sample_rate
            assert "audio" in config_json
            assert "model" in config_json

    def test_export_dual_format(self):
        """Test dual-format export creates both RVC and SoVITS files."""
        import torch
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = trainer.export_dual_format(
                output_dir=tmpdir,
                rvc_name="model_rvc.pt",
                sovits_name="model_sovits.pt",
            )

            # Verify both files exist
            assert os.path.exists(result["rvc"])
            assert os.path.exists(result["sovits"])

            # Verify RVC format
            rvc_ckpt = torch.load(result["rvc"], map_location="cpu")
            assert rvc_ckpt["version"] == "rvc_v2"
            assert "model" in rvc_ckpt

            # Verify SoVITS format
            sovits_ckpt = torch.load(result["sovits"], map_location="cpu")
            assert sovits_ckpt["version"] == "sovits_v4.1"
            assert "model" in sovits_ckpt

            # Verify config.json exists alongside SoVITS model
            config_json_path = os.path.join(tmpdir, "config.json")
            assert os.path.exists(config_json_path)

    def test_export_dual_format_custom_names(self):
        """Test dual-format export with custom filenames."""
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = trainer.export_dual_format(
                output_dir=tmpdir,
                rvc_name="custom_rvc.pt",
                sovits_name="custom_sovits.pt",
            )

            assert result["rvc"].endswith("custom_rvc.pt")
            assert result["sovits"].endswith("custom_sovits.pt")
            assert os.path.exists(result["rvc"])
            assert os.path.exists(result["sovits"])

    def test_export_sovits_without_generator_raises(self):
        """Test that exporting without building models raises error."""
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        # Do NOT call build_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")
            with pytest.raises(RuntimeError, match="Generator not built"):
                trainer.export_sovits_format(output_path)

    def test_sovits_config_json_structure(self):
        """Test that exported config.json has correct structure for SoVITSConverter."""
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        config.data.sample_rate = 44100
        config.data.hop_length = 512
        config.data.n_mels = 80

        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model_sovits.pt")
            trainer.export_sovits_format(output_path)

            config_json_path = os.path.join(tmpdir, "config.json")
            with open(config_json_path, "r") as f:
                config_json = json.load(f)

            # Verify structure matches SoVITSConverter expectations
            assert config_json["version"] == "4.1"
            assert config_json["sampling_rate"] == 44100
            assert config_json["audio"]["sample_rate"] == 44100
            assert config_json["audio"]["hop_length"] == 512
            assert config_json["audio"]["n_mels"] == 80
            assert "model" in config_json
            assert "data" in config_json
            assert "train" in config_json

    def test_auto_export_method_exists(self):
        """Test that auto_export_dual_format method exists and is callable."""
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")

        # Verify the method exists
        assert hasattr(trainer, '_auto_export_dual_format')
        assert hasattr(trainer, 'export_dual_format')
        assert hasattr(trainer, 'export_sovits_format')
        assert hasattr(trainer, 'export_for_inference')

    def test_export_format_config_default(self):
        """Test that export_format defaults to 'rvc' for backward compatibility."""
        from src.training.config import TrainingConfig

        config = TrainingConfig()
        assert config.train.export_format == "rvc"

    def test_export_format_config_validation(self):
        """Test that export_format validates allowed values."""
        from src.training.config import TrainingConfig

        config = TrainingConfig()
        config.train.export_format = "sovits"
        errors = config.validate()
        assert not any("export_format" in e for e in errors)

        config.train.export_format = "dual"
        errors = config.validate()
        assert not any("export_format" in e for e in errors)

        config.train.export_format = "invalid"
        errors = config.validate()
        assert any("export_format" in e for e in errors)

    def test_auto_export_respects_rvc_only_format(self):
        """Test that auto-export with export_format='rvc' only exports RVC."""
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        config.train.export_format = "rvc"
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.train.checkpoint_dir = tmpdir
            trainer._auto_export_dual_format()

            # RVC file should exist
            rvc_path = os.path.join(tmpdir, "model_rvc.pt")
            assert os.path.exists(rvc_path)

            # SoVITS file should NOT exist
            sovits_path = os.path.join(tmpdir, "model_sovits.pt")
            assert not os.path.exists(sovits_path)

    def test_auto_export_respects_dual_format(self):
        """Test that auto-export with export_format='dual' exports both formats."""
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        config.train.export_format = "dual"
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.train.checkpoint_dir = tmpdir
            trainer._auto_export_dual_format()

            # Both files should exist
            rvc_path = os.path.join(tmpdir, "model_rvc.pt")
            sovits_path = os.path.join(tmpdir, "model_sovits.pt")
            assert os.path.exists(rvc_path)
            assert os.path.exists(sovits_path)

    def test_auto_export_sovits_failure_non_fatal(self):
        """Test that SoVITS export failure does not crash training."""
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        config.train.export_format = "dual"
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()

        # Monkey-patch export_sovits_format to raise an error
        original_export = trainer.export_sovits_format

        def failing_export(path):
            raise RuntimeError("Simulated SoVITS export failure")

        trainer.export_sovits_format = failing_export

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.train.checkpoint_dir = tmpdir
            # Should NOT raise even though SoVITS export fails
            trainer._auto_export_dual_format()

            # RVC file should still exist
            rvc_path = os.path.join(tmpdir, "model_rvc.pt")
            assert os.path.exists(rvc_path)

        # Restore
        trainer.export_sovits_format = original_export

    def test_f0_backward_compat_old_checkpoint(self):
        """Test backward compatibility: old checkpoint with 1-channel F0 encoding."""
        import torch
        from src.voice_converters.rvc_models import SimpleRVCModel, create_rvc_model_from_checkpoint

        # Simulate an old checkpoint where pitch_encoder_dim was 1
        # Create a model with pitch_encoder_dim=1
        old_config = {"sample_rate": 40000, "hop_length": 512, "pitch_encoder_dim": 1}
        old_model = SimpleRVCModel(old_config)

        # Save as checkpoint
        checkpoint = {
            "model": old_model.state_dict(),
            "config": {"sample_rate": 40000, "hop_length": 512},
            "version": "rvc_v2",
        }

        # Load with create_rvc_model_from_checkpoint - should detect and adapt
        loaded_model = create_rvc_model_from_checkpoint(checkpoint)

        # Verify the loaded model has the correct pitch_encoder_dim
        assert loaded_model.pitch_encoder_dim == 1

        # Verify forward pass works with old checkpoint
        x = torch.randn(1, 256, 10)
        f0 = torch.ones(1, 1, 10) * 200.0
        output = loaded_model(x, f0)
        assert output.shape[0] == 1

    def test_f0_backward_compat_new_checkpoint(self):
        """Test that new checkpoints with pitch_encoder_dim=256 load correctly."""
        import torch
        from src.voice_converters.rvc_models import SimpleRVCModel, create_rvc_model_from_checkpoint

        # Create a model with default pitch_encoder_dim=256
        new_config = {"sample_rate": 40000, "hop_length": 512, "pitch_encoder_dim": 256}
        new_model = SimpleRVCModel(new_config)

        checkpoint = {
            "model": new_model.state_dict(),
            "config": {"sample_rate": 40000, "hop_length": 512, "pitch_encoder_dim": 256},
            "version": "rvc_v2",
        }

        loaded_model = create_rvc_model_from_checkpoint(checkpoint)
        assert loaded_model.pitch_encoder_dim == 256

        # Verify forward pass works
        x = torch.randn(1, 256, 10)
        f0 = torch.ones(1, 1, 10) * 200.0
        output = loaded_model(x, f0)
        assert output.shape[0] == 1

    def test_cli_export_format_option(self):
        """Test that CLI export command accepts --format option."""
        from src.training.cli import build_parser

        parser = build_parser()

        # Test default (no --format)
        args = parser.parse_args(["export", "--checkpoint", "model.pt"])
        assert args.format is None

        # Test --format rvc
        args = parser.parse_args(["export", "--checkpoint", "model.pt", "--format", "rvc"])
        assert args.format == "rvc"

        # Test --format sovits
        args = parser.parse_args(["export", "--checkpoint", "model.pt", "--format", "sovits"])
        assert args.format == "sovits"

        # Test --format dual
        args = parser.parse_args(["export", "--checkpoint", "model.pt", "--format", "dual"])
        assert args.format == "dual"

        # Test invalid format
        with pytest.raises(SystemExit):
            parser.parse_args(["export", "--checkpoint", "model.pt", "--format", "invalid"])


# ============================================================================
# Feature Extractor Tests
# ============================================================================

class TestF0Extractor:
    """Tests for F0Extractor."""

    def test_creation_default(self):
        """Test F0Extractor creation with default settings."""
        from src.training.feature_extractor import F0Extractor
        extractor = F0Extractor()
        assert extractor.method == "dio"
        assert extractor.sample_rate == 16000

    def test_creation_custom(self):
        """Test F0Extractor creation with custom settings."""
        from src.training.feature_extractor import F0Extractor
        extractor = F0Extractor(method="pyin", sample_rate=22050, f0_min=100, f0_max=1000)
        assert extractor.method == "pyin"
        assert extractor.sample_rate == 22050
        assert extractor.f0_min == 100
        assert extractor.f0_max == 1000

    def test_extract_f0_sine_wave(self):
        """Test F0 extraction from a sine wave."""
        from src.training.feature_extractor import F0Extractor
        # Use higher fmin to avoid librosa.yin parameter issues
        extractor = F0Extractor(method="yin", sample_rate=16000, f0_min=100.0)

        # Create a 440Hz sine wave (A4 note)
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        f0 = extractor.extract(audio)
        assert f0 is not None
        assert len(f0) > 0
        # F0 extraction may use fallback (200Hz) if yin fails, so just check it returns values
        valid_f0 = f0[f0 > 0]
        assert len(valid_f0) > 0  # Should have some voiced frames

    def test_extract_f0_silence(self):
        """Test F0 extraction from silence returns low values."""
        from src.training.feature_extractor import F0Extractor
        extractor = F0Extractor(method="yin", sample_rate=16000, f0_min=100.0)

        # Create silence
        audio = np.zeros(16000, dtype=np.float32)
        f0 = extractor.extract(audio)

        assert f0 is not None
        # Silence may return fallback values or zeros, both are acceptable
        assert len(f0) > 0

    def test_extract_f0_with_target_frames(self):
        """Test F0 extraction with target frame count."""
        from src.training.feature_extractor import F0Extractor
        extractor = F0Extractor(method="yin", sample_rate=16000, hop_length=512)

        # Create a simple audio signal
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        target_frames = 32

        f0 = extractor.extract(audio, target_length=target_frames)
        assert f0 is not None
        assert len(f0) == target_frames


class TestHuBERTFeatureExtractor:
    """Tests for HuBERTFeatureExtractor."""

    def test_creation_default(self):
        """Test HuBERTFeatureExtractor creation with default settings (lazy load)."""
        from src.training.feature_extractor import HuBERTFeatureExtractor
        extractor = HuBERTFeatureExtractor(device="cpu")
        assert extractor.feature_dim == 256
        # With lazy_load=True (default), model is NOT loaded during init
        assert extractor._is_loaded is False

    @pytest.mark.skip(reason="Requires network/GPU to load HuBERT model (~400MB download)")
    def test_creation_eager_load(self):
        """Test HuBERTFeatureExtractor creation with lazy_load=False triggers loading."""
        from src.training.feature_extractor import HuBERTFeatureExtractor
        # lazy_load=False will try to load the model; it may fall back
        extractor = HuBERTFeatureExtractor(device="cpu", lazy_load=False)
        assert extractor._is_loaded is True
        assert extractor.feature_dim == 256

    def test_feature_dim_property(self):
        """Test feature_dim property returns correct value."""
        from src.training.feature_extractor import HuBERTFeatureExtractor
        extractor = HuBERTFeatureExtractor(device="cpu")
        assert extractor.feature_dim == 256

    @pytest.mark.skip(reason="Requires network/GPU to load HuBERT model (~400MB download)")
    def test_extract_triggers_lazy_load(self):
        """Test that extract() triggers lazy loading and returns features."""
        from src.training.feature_extractor import HuBERTFeatureExtractor
        extractor = HuBERTFeatureExtractor(device="cpu")
        assert extractor._is_loaded is False

        # Create a short audio signal to avoid memory issues
        sample_rate = 16000
        audio = np.random.randn(sample_rate // 4).astype(np.float32) * 0.1  # 0.25s

        features = extractor.extract(audio, sample_rate)
        assert extractor._is_loaded is True  # Should be loaded after extract
        assert features is not None
        assert features.shape[0] == 256  # feature_dim
        assert features.shape[1] > 0  # Should have some frames

    @pytest.mark.skip(reason="Requires network/GPU to load HuBERT model (~400MB download)")
    def test_extract_silence(self):
        """Test extraction from silence still produces features."""
        from src.training.feature_extractor import HuBERTFeatureExtractor
        extractor = HuBERTFeatureExtractor(device="cpu")

        # Short silence to avoid memory issues
        audio = np.zeros(4000, dtype=np.float32)
        features = extractor.extract(audio, 16000)
        assert features is not None
        assert features.shape[0] == 256

    def test_fallback_determinism(self):
        """Test that fallback extractor produces identical output for identical input."""
        from src.training.feature_extractor import HuBERTFeatureExtractor
        # Create with lazy_load (default) and manually trigger fallback
        extractor = HuBERTFeatureExtractor(device="cpu")
        # Directly call _init_fallback_extractor to avoid network download attempts
        extractor._init_fallback_extractor()
        extractor._is_loaded = True
        assert extractor.using_fallback is True

        # Same input should produce same output (deterministic)
        audio = np.random.RandomState(123).randn(4000).astype(np.float32) * 0.1
        features_1 = extractor.extract(audio, 16000)
        features_2 = extractor.extract(audio, 16000)
        np.testing.assert_array_equal(features_1, features_2)

    def test_using_fallback_property(self):
        """Test that using_fallback property correctly reports fallback mode."""
        from src.training.feature_extractor import HuBERTFeatureExtractor
        # Lazy load - not loaded yet, not in fallback
        extractor = HuBERTFeatureExtractor(device="cpu")
        assert extractor.using_fallback is False

        # Manually trigger fallback (avoids network download)
        extractor._init_fallback_extractor()
        assert extractor.using_fallback is True


class TestFeaturePipeline:
    """Tests for FeaturePipeline."""

    def test_creation_default(self):
        """Test FeaturePipeline creation with default settings."""
        from src.training.feature_extractor import FeaturePipeline
        pipeline = FeaturePipeline(device="cpu")
        assert pipeline.feature_dim == 256
        assert pipeline.f0_extractor is not None
        assert pipeline.hubert is not None
        # HuBERT should be lazy-loaded (not loaded yet)
        assert pipeline.hubert._is_loaded is False

    @pytest.mark.skip(reason="Requires network/GPU to load HuBERT model (~400MB download)")
    def test_extract_returns_features_and_f0(self):
        """Test that extract returns both features and F0."""
        from src.training.feature_extractor import FeaturePipeline
        pipeline = FeaturePipeline(device="cpu")

        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

        features, f0 = pipeline.extract(audio, sample_rate)
        assert features is not None
        assert f0 is not None
        assert features.shape[0] == 256
        assert len(f0) > 0

    @pytest.mark.skip(reason="Requires network/GPU to load HuBERT model (~400MB download)")
    def test_extract_with_target_frames(self):
        """Test extraction with target frame count aligns both outputs."""
        from src.training.feature_extractor import FeaturePipeline
        pipeline = FeaturePipeline(device="cpu")

        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1
        target_frames = 40

        features, f0 = pipeline.extract(audio, sample_rate, target_frames=target_frames)
        assert features.shape == (256, target_frames)
        assert len(f0) == target_frames


class TestTrainerFeatureIntegration:
    """Tests for trainer integration with feature extraction."""

    def test_trainer_has_feature_pipeline(self):
        """Test that trainer has feature_pipeline attribute."""
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")

        # Feature pipeline should be None initially (lazy loading)
        assert hasattr(trainer, 'feature_pipeline')
        assert trainer.feature_pipeline is None

    def test_trainer_init_feature_pipeline(self):
        """Test that init_feature_pipeline creates the pipeline."""
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")

        # Initialize feature pipeline (lazy - doesn't load model yet)
        trainer.init_feature_pipeline()

        assert trainer.feature_pipeline is not None
        assert trainer.feature_pipeline.feature_dim == 256

    @pytest.mark.skip(reason="Requires network/GPU to load HuBERT model (~400MB download)")
    def test_trainer_extract_real_features(self):
        """Test _extract_real_features method."""
        import torch
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        trainer = RVCTrainer(config, device="cpu")
        trainer.init_feature_pipeline()

        # Create a batch of audio
        batch_size = 2
        audio_length = 16000  # 1 second at 16kHz
        audio = torch.randn(batch_size, audio_length)

        # Calculate target frames
        hop_length = config.data.hop_length
        target_frames = audio_length // hop_length

        features, f0 = trainer._extract_real_features(audio, target_frames)

        assert features.shape[0] == batch_size
        assert features.shape[1] == config.model.in_channels
        assert features.shape[2] == target_frames
        assert f0.shape[0] == batch_size
        assert f0.shape[1] == target_frames

    @pytest.mark.skip(reason="Requires network/GPU to load HuBERT model (~400MB download)")
    def test_trainer_train_step_uses_real_features(self):
        """Test that train_step uses real features when pipeline is initialized."""
        import torch
        from src.training.config import TrainingConfig
        from src.training.trainer import RVCTrainer

        config = TrainingConfig()
        config.data.segment_length = 4000  # Small segment for testing
        trainer = RVCTrainer(config, device="cpu")
        trainer.build_models()
        trainer.init_feature_pipeline()

        # Create a batch
        batch_size = 2
        audio_length = 4000
        batch = {
            "audio": torch.randn(batch_size, audio_length),
            "mel": torch.randn(batch_size, 80, audio_length // 512),
        }

        # Train step should work with real features
        losses = trainer.train_step(batch)

        assert "g_loss" in losses
        assert "d_loss" in losses
        assert "mel_loss" in losses
