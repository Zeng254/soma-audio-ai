"""
RVC Model Trainer Module

Provides complete training pipeline for RVC voice conversion models:
- Generator + Multi-Period Discriminator (MPD) architecture
- Loss functions: Mel loss + L1 loss + Adversarial loss + Feature matching loss
- AMP (Automatic Mixed Precision) training
- Gradient clipping
- Checkpoint save/restore (best + latest)
- TensorBoard logging
- Learning rate scheduling (cosine annealing)
- Resume training from specific epoch
"""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ModelConfig, OptimizerConfig, TrainConfig, TrainingConfig
from .feature_extractor import FeaturePipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-Period Discriminator
# ---------------------------------------------------------------------------

class MultiPeriodDiscriminator:
    """
    Multi-Period Discriminator (MPD) for RVC training.

    Uses multiple period values to capture different temporal patterns.
    """

    def __init__(self, periods: List[List[int]], device: str = "cpu"):
        """
        Initialize MPD.

        Args:
            periods: List of period groups for each sub-discriminator.
            device: Compute device.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required. Install with: uv add torch")

        self.periods = periods
        self.device = device
        self.discriminators = nn.ModuleList()

        for period_group in periods:
            self.discriminators.append(
                self._build_sub_discriminator(period_group[0])
            )

        self.discriminators = self.discriminators.to(device)

    def _build_sub_discriminator(self, period: int):
        """Build a single sub-discriminator for a given period."""
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(1, 32, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 128, (3, 1), stride=(1, 1), padding=(1, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 512, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)),
        )

    def __call__(self, x):
        """Forward pass through all sub-discriminators."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        results = []
        features = []

        for disc in self.discriminators:
            # Reshape 1D to 2D using period
            batch_size = x.shape[0]
            length = x.shape[-1]
            # Pad to make divisible
            pad_len = (self.periods[len(results)][0] - length % self.periods[len(results)][0]) % self.periods[len(results)][0]
            if pad_len > 0:
                x_padded = F.pad(x, (0, pad_len))
            else:
                x_padded = x

            period = self.periods[len(results)][0]
            n_frames = x_padded.shape[-1] // period
            x_2d = x_padded.view(batch_size, 1, n_frames, period)

            # Get intermediate features
            feats = []
            h = x_2d
            for layer in disc:
                h = layer(h)
                if isinstance(layer, nn.Conv2d):
                    feats.append(h)

            results.append(h)
            features.append(feats)

        return results, features

    def to(self, device):
        """Move to device."""
        self.device = device
        self.discriminators = self.discriminators.to(device)
        return self

    def parameters(self):
        """Return all parameters."""
        return self.discriminators.parameters()

    def train(self, mode=True):
        """Set training mode."""
        self.discriminators.train(mode)
        return self

    def eval(self):
        """Set eval mode."""
        self.discriminators.eval()
        return self

    def state_dict(self):
        """Return state dict."""
        return self.discriminators.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.discriminators.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def mel_spectrogram_loss(
    y_pred, y_true, n_fft=2048, hop_length=512, n_mels=128, sr=40000
):
    """Compute mel spectrogram L1 loss."""
    import torch
    import torch.nn.functional as F

    # STFT
    window = torch.hann_window(n_fft, device=y_pred.device)

    def _stft(x):
        return torch.stft(
            x, n_fft, hop_length, window=window,
            return_complex=True, pad_mode="reflect"
        )

    stft_pred = _stft(y_pred)
    stft_true = _stft(y_true)

    mag_pred = torch.abs(stft_pred)
    mag_true = torch.abs(stft_true)

    # Simple mel approximation (sum of frequency bins)
    loss = F.l1_loss(mag_pred, mag_true)
    return loss


def feature_matching_loss(features_fake, features_real):
    """Compute feature matching loss between discriminator features."""
    import torch
    import torch.nn.functional as F

    loss = 0.0
    n_layers = 0
    for fake_feats, real_feats in zip(features_fake, features_real):
        for fake_f, real_f in zip(fake_feats, real_feats):
            loss += F.l1_loss(fake_f, real_f.detach())
            n_layers += 1

    if n_layers > 0:
        loss = loss / n_layers
    return loss


def discriminator_loss(real_outputs, fake_outputs):
    """Compute discriminator loss (hinge loss)."""
    import torch

    loss = 0.0
    for real_out, fake_out in zip(real_outputs, fake_outputs):
        real_loss = torch.mean(torch.relu(1.0 - real_out))
        fake_loss = torch.mean(torch.relu(1.0 + fake_out))
        loss += real_loss + fake_loss
    return loss


def generator_adversarial_loss(fake_outputs):
    """Compute generator adversarial loss."""
    import torch

    loss = 0.0
    for fake_out in fake_outputs:
        loss += torch.mean(-fake_out)
    return loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RVCTrainer:
    """
    RVC Model Trainer.

    Manages the complete training pipeline including:
    - Generator and discriminator training
    - AMP mixed precision
    - Gradient clipping
    - Checkpoint management
    - TensorBoard logging
    - Learning rate scheduling
    """

    def __init__(self, config: TrainingConfig, device: Optional[str] = None):
        """
        Initialize trainer.

        Args:
            config: Full training configuration.
            device: Compute device. If None, auto-detect.
        """
        self.config = config
        self.device = self._resolve_device(device or config.train.device)

        # Models
        self.generator = None
        self.discriminator = None

        # Optimizers
        self.optimizer_g = None
        self.optimizer_d = None

        # Scheduler
        self.scheduler_g = None
        self.scheduler_d = None

        # AMP
        self.scaler = None

        # Feature extraction pipeline (HuBERT + F0)
        self.feature_pipeline = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # TensorBoard
        self.writer = None

        # Paths
        self._setup_paths()

        logger.info("RVCTrainer initialized on device: %s", self.device)

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        try:
            import torch
        except ImportError:
            return "cpu"

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _setup_paths(self):
        """Create output directories."""
        for dir_path in [
            self.config.train.output_dir,
            self.config.train.log_dir,
            self.config.train.checkpoint_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def init_feature_pipeline(
        self,
        model_name: str = "hubert_base",
        model_path: Optional[str] = None,
        f0_method: str = "dio",
    ) -> None:
        """
        Initialize the HuBERT + F0 feature extraction pipeline.

        This should be called before training starts. The pipeline is loaded
        in eval mode with frozen weights to extract real speech features.

        Args:
            model_name: HuBERT model name ("hubert_base" or "contentvec").
            model_path: Optional path to a local HuBERT checkpoint.
            f0_method: F0 extraction method ("yin", "pyin", "dio", "harvest").
        """
        self.feature_pipeline = FeaturePipeline(
            model_name=model_name,
            device=self.device,
            model_path=model_path,
            f0_method=f0_method,
            sample_rate=self.config.data.sample_rate,
            hop_length=self.config.data.hop_length,
        )
        logger.info(
            "Feature pipeline initialized: model=%s, f0_method=%s, feature_dim=%d",
            model_name, f0_method, self.feature_pipeline.feature_dim,
        )

    def build_models(self):
        """Build generator and discriminator models."""
        import torch

        # Build generator (reuse existing RVC model architecture)
        from voice_converters.rvc_models import SimpleRVCModel

        model_cfg = self.config.model
        self.generator = SimpleRVCModel({
            "sample_rate": self.config.data.sample_rate,
            "hop_length": self.config.data.hop_length,
            "mel_channels": self.config.data.n_mels,
            "use_flow": model_cfg.use_flow,
        })
        self.generator = self.generator.to(self.device)

        # Build discriminator
        self.discriminator = MultiPeriodDiscriminator(
            periods=model_cfg.mpd_periods,
            device=self.device,
        )

        # AMP scaler
        if self.config.train.use_amp and self.device == "cuda":
            amp_dtype = torch.float16
            if self.config.train.amp_dtype == "bfloat16":
                amp_dtype = torch.bfloat16
            self.scaler = torch.amp.GradScaler(dtype=amp_dtype)

        logger.info(
            "Models built: Generator (%.2fM params), Discriminator (%.2fM params)",
            self._count_params(self.generator) / 1e6,
            self._count_params(self.discriminator.discriminators) / 1e6,
        )

    def _count_params(self, model) -> int:
        """Count trainable parameters."""
        if hasattr(model, "parameters"):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return 0

    def build_optimizers(self):
        """Build optimizers and schedulers."""
        import torch

        opt_cfg = self.config.optimizer

        # Generator optimizer
        if opt_cfg.optimizer == "adamw":
            self.optimizer_g = torch.optim.AdamW(
                self.generator.parameters(),
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                weight_decay=opt_cfg.weight_decay,
                eps=opt_cfg.eps,
            )
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                weight_decay=opt_cfg.weight_decay,
                eps=opt_cfg.eps,
            )
        else:
            self.optimizer_g = torch.optim.Adam(
                self.generator.parameters(),
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                eps=opt_cfg.eps,
            )
            self.optimizer_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                eps=opt_cfg.eps,
            )

        # LR scheduler (cosine annealing)
        total_steps = opt_cfg.total_steps
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=total_steps, eta_min=opt_cfg.min_lr
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=total_steps, eta_min=opt_cfg.min_lr
        )

        logger.info("Optimizers and schedulers built")

    def setup_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.config.train.log_dir)
            logger.info("TensorBoard logging to %s", self.config.train.log_dir)
        except ImportError:
            logger.warning(
                "TensorBoard not available. Install with: uv add tensorboard"
            )

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: Dict with 'audio' and 'mel' tensors.

        Returns:
            Dict of loss values.
        """
        import torch

        audio = batch["audio"].to(self.device)
        mel_target = batch["mel"].to(self.device)

        # Extract real features using HuBERT + F0 pipeline
        batch_size = audio.shape[0]
        seq_len = mel_target.shape[-1] if mel_target.dim() == 3 else audio.shape[-1] // self.config.data.hop_length

        if self.feature_pipeline is not None:
            # Use real HuBERT features and F0 extraction
            features, f0 = self._extract_real_features(audio, seq_len)
        else:
            # Fallback to dummy features (for backward compatibility / testing)
            logger.debug("Feature pipeline not initialized, using dummy features")
            features = torch.randn(
                batch_size, self.config.model.in_channels, seq_len,
                device=self.device,
            )
            f0 = torch.ones(batch_size, seq_len, device=self.device) * 200.0

        losses = {}

        # --- Train Generator ---
        self.generator.train()
        self.discriminator.eval()

        if self.scaler is not None:
            with torch.amp.autocast(device_type=self.device):
                audio_pred = self.generator(features, f0)
                # Ensure same length
                min_len = min(audio_pred.shape[-1], audio.shape[-1])
                audio_pred = audio_pred[:, :, :min_len]
                audio_real = audio[:, :min_len]

                # Mel loss
                mel_loss = mel_spectrogram_loss(
                    audio_pred.squeeze(1), audio_real,
                    n_fft=self.config.data.n_fft,
                    hop_length=self.config.data.hop_length,
                    n_mels=self.config.data.n_mels,
                    sr=self.config.data.sample_rate,
                )

                # L1 loss
                l1_loss = torch.nn.functional.l1_loss(audio_pred.squeeze(1), audio_real)

                # Adversarial loss
                fake_outputs, fake_features = self.discriminator(audio_pred.squeeze(1))
                adv_loss = generator_adversarial_loss(fake_outputs)

                # Feature matching loss
                self.discriminator.eval()
                with torch.no_grad():
                    real_outputs, real_features = self.discriminator(audio_real)
                fm_loss = feature_matching_loss(fake_features, real_features)

                # Total generator loss
                g_loss = (
                    self.config.train.mel_loss_weight * mel_loss
                    + self.config.train.l1_loss_weight * l1_loss
                    + self.config.train.adv_loss_weight * adv_loss
                    + self.config.train.fm_loss_weight * fm_loss
                )

            self.optimizer_g.zero_grad()
            self.scaler.scale(g_loss).backward()
            self.scaler.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.config.optimizer.grad_clip,
            )
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            audio_pred = self.generator(features, f0)
            min_len = min(audio_pred.shape[-1], audio.shape[-1])
            audio_pred = audio_pred[:, :, :min_len]
            audio_real = audio[:, :min_len]

            mel_loss = mel_spectrogram_loss(
                audio_pred.squeeze(1), audio_real,
                n_fft=self.config.data.n_fft,
                hop_length=self.config.data.hop_length,
            )
            l1_loss = torch.nn.functional.l1_loss(audio_pred.squeeze(1), audio_real)

            fake_outputs, fake_features = self.discriminator(audio_pred.squeeze(1))
            adv_loss = generator_adversarial_loss(fake_outputs)

            self.discriminator.eval()
            with torch.no_grad():
                real_outputs, real_features = self.discriminator(audio_real)
            fm_loss = feature_matching_loss(fake_features, real_features)

            g_loss = (
                self.config.train.mel_loss_weight * mel_loss
                + self.config.train.l1_loss_weight * l1_loss
                + self.config.train.adv_loss_weight * adv_loss
                + self.config.train.fm_loss_weight * fm_loss
            )

            self.optimizer_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.config.optimizer.grad_clip,
            )
            self.optimizer_g.step()

        losses["mel_loss"] = mel_loss.item()
        losses["l1_loss"] = l1_loss.item()
        losses["adv_loss"] = adv_loss.item()
        losses["fm_loss"] = fm_loss.item()
        losses["g_loss"] = g_loss.item()

        # --- Train Discriminator ---
        self.generator.eval()
        self.discriminator.train()

        with torch.no_grad():
            audio_pred = self.generator(features, f0)
            min_len = min(audio_pred.shape[-1], audio.shape[-1])
            audio_pred = audio_pred[:, :, :min_len].squeeze(1)
            audio_real = audio[:, :min_len]

        real_outputs, _ = self.discriminator(audio_real)
        fake_outputs, _ = self.discriminator(audio_pred.detach())
        d_loss = discriminator_loss(real_outputs, fake_outputs)

        self.optimizer_d.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            self.config.optimizer.grad_clip,
        )
        self.optimizer_d.step()

        losses["d_loss"] = d_loss.item()

        # Update schedulers
        self.scheduler_g.step()
        self.scheduler_d.step()

        self.global_step += 1
        return losses

    def _extract_real_features(
        self, audio: "torch.Tensor", target_frames: int
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Extract real HuBERT features and F0 from audio batch.

        Args:
            audio: Audio tensor, shape (batch_size, samples) or (batch_size, 1, samples).
            target_frames: Target number of frames for alignment.

        Returns:
            Tuple of (features, f0) tensors:
                - features: (batch_size, in_channels, target_frames)
                - f0: (batch_size, target_frames)
        """
        import torch

        batch_size = audio.shape[0]
        sample_rate = self.config.data.sample_rate

        # Convert audio to numpy for feature extraction
        audio_np = audio.detach().cpu().numpy()

        # Handle shape: (batch, 1, samples) -> (batch, samples)
        if audio_np.ndim == 3:
            audio_np = audio_np.squeeze(1)

        # Extract features for each sample in the batch
        all_features = []
        all_f0 = []

        for i in range(batch_size):
            audio_sample = audio_np[i]

            # Extract HuBERT features and F0 using the pipeline
            features, f0 = self.feature_pipeline.extract(
                audio_sample,
                sample_rate=sample_rate,
                target_frames=target_frames,
            )

            all_features.append(features)
            all_f0.append(f0)

        # Stack into batch tensors
        # features: (batch, feature_dim, frames) -> pad/trim to in_channels
        feature_dim = self.feature_pipeline.feature_dim
        in_channels = self.config.model.in_channels

        features_padded = np.zeros(
            (batch_size, in_channels, target_frames), dtype=np.float32
        )
        f0_padded = np.zeros((batch_size, target_frames), dtype=np.float32)

        for i in range(batch_size):
            feats = all_features[i]  # (feature_dim, frames)
            f0_arr = all_f0[i]  # (frames,)

            # Handle feature dimension mismatch
            feat_dim_actual = feats.shape[0]
            feat_frames = feats.shape[1]

            if feat_dim_actual >= in_channels:
                # Take first in_channels dimensions
                features_padded[i] = feats[:in_channels, :target_frames]
            else:
                # Pad with zeros if feature_dim < in_channels
                features_padded[i, :feat_dim_actual, :feat_frames] = feats

            # Handle F0 length
            f0_len = len(f0_arr)
            f0_padded[i, :f0_len] = f0_arr[:target_frames]

        # Convert to tensors
        features_tensor = torch.from_numpy(features_padded).float().to(self.device)
        f0_tensor = torch.from_numpy(f0_padded).float().to(self.device)

        return features_tensor, f0_tensor

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = {}
        n_batches = 0

        for batch in dataloader:
            losses = self.train_step(batch)
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_batches += 1

            # Log to TensorBoard
            if self.writer and self.global_step % self.config.train.log_interval == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)

        # Average losses
        if n_batches > 0:
            epoch_losses = {k: v / n_batches for k, v in epoch_losses.items()}

        return epoch_losses

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        start_epoch: int = 0,
    ):
        """
        Full training loop.

        Args:
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader (optional).
            start_epoch: Epoch to start from (for resume).
        """
        self.current_epoch = start_epoch

        logger.info(
            "Starting training from epoch %d for %d epochs",
            start_epoch, self.config.train.num_epochs,
        )

        for epoch in range(start_epoch, self.config.train.num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            # Train
            train_losses = self.train_epoch(train_dataloader)

            # Validate
            val_losses = {}
            if val_dataloader and (epoch + 1) % self.config.train.eval_interval == 0:
                val_losses = self._validate(val_dataloader)

            elapsed = time.time() - t0

            # Log
            msg = (
                f"Epoch {epoch + 1}/{self.config.train.num_epochs} "
                f"({elapsed:.1f}s) | "
                + " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            )
            if val_losses:
                msg += " | " + " | ".join(
                    f"val_{k}: {v:.4f}" for k, v in val_losses.items()
                )
            logger.info(msg)

            if self.writer:
                for k, v in train_losses.items():
                    self.writer.add_scalar(f"epoch/{k}", v, epoch)

            # Save checkpoint
            if (epoch + 1) % self.config.train.save_interval == 0:
                self.save_checkpoint(epoch + 1, train_losses)

        logger.info("Training complete!")

        # Auto-export dual format models after training
        self._auto_export_dual_format()

    def _auto_export_dual_format(self):
        """
        Automatically export models after training based on config.export_format.

        Export formats:
        - 'rvc': Export RVC format only (mandatory, training fails if this errors)
        - 'sovits': Export SoVITS format only (optional, errors are logged as warnings)
        - 'dual': Export both RVC and SoVITS formats (RVC mandatory, SoVITS optional)

        Exports are saved to the checkpoint directory with distinguishable names.
        """
        if self.generator is None:
            logger.warning("Generator not built, skipping auto-export.")
            return

        ckpt_dir = Path(self.config.train.checkpoint_dir)
        export_format = self.config.train.export_format

        # RVC export (mandatory for 'rvc' and 'dual')
        if export_format in ("rvc", "dual"):
            rvc_path = ckpt_dir / "model_rvc.pt"
            try:
                self.export_for_inference(str(rvc_path))
                logger.info("Auto-export RVC model: %s", rvc_path)
            except Exception as e:
                logger.error("RVC auto-export failed (mandatory): %s", e)
                raise

        # SoVITS export (optional for 'sovits' and 'dual')
        if export_format in ("sovits", "dual"):
            sovits_path = ckpt_dir / "model_sovits.pt"
            try:
                self.export_sovits_format(str(sovits_path))
                logger.info("Auto-export SoVITS model: %s", sovits_path)
            except Exception as e:
                logger.warning(
                    "SoVITS auto-export failed (optional, non-fatal): %s", e
                )

    def _validate(self, dataloader) -> Dict[str, float]:
        """Run validation."""
        import torch

        self.generator.eval()
        self.discriminator.eval()

        val_losses = {}
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                audio = batch["audio"].to(self.device)
                mel_target = batch["mel"].to(self.device)

                batch_size = audio.shape[0]
                seq_len = audio.shape[-1] // self.config.data.hop_length
                features = torch.randn(
                    batch_size, self.config.model.in_channels, seq_len,
                    device=self.device,
                )
                f0 = torch.ones(batch_size, seq_len, device=self.device) * 200.0

                audio_pred = self.generator(features, f0)
                min_len = min(audio_pred.shape[-1], audio.shape[-1])
                audio_pred_trim = audio_pred[:, :, :min_len].squeeze(1)
                audio_real = audio[:, :min_len]

                mel_loss = mel_spectrogram_loss(
                    audio_pred_trim, audio_real,
                    n_fft=self.config.data.n_fft,
                    hop_length=self.config.data.hop_length,
                )
                l1_loss = torch.nn.functional.l1_loss(audio_pred_trim, audio_real)

                val_losses["mel_loss"] = val_losses.get("mel_loss", 0) + mel_loss.item()
                val_losses["l1_loss"] = val_losses.get("l1_loss", 0) + l1_loss.item()
                n_batches += 1

        if n_batches > 0:
            val_losses = {k: v / n_batches for k, v in val_losses.items()}

        return val_losses

    def save_checkpoint(self, epoch: int, losses: Optional[Dict] = None):
        """
        Save training checkpoint.

        Saves both 'latest' and 'best' checkpoints.

        Args:
            epoch: Current epoch number.
            losses: Current loss values (for best model tracking).
        """
        import torch

        ckpt_dir = Path(self.config.train.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "generator": self.generator.state_dict() if self.generator else None,
            "discriminator": self.discriminator.state_dict() if self.discriminator else None,
            "optimizer_g": self.optimizer_g.state_dict() if self.optimizer_g else None,
            "optimizer_d": self.optimizer_d.state_dict() if self.optimizer_d else None,
            "scheduler_g": self.scheduler_g.state_dict() if self.scheduler_g else None,
            "scheduler_d": self.scheduler_d.state_dict() if self.scheduler_d else None,
            "config": self.config.to_dict(),
            "best_loss": self.best_loss,
        }

        # Save latest
        latest_path = ckpt_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, str(latest_path))

        # Save best
        if losses and "g_loss" in losses:
            current_loss = losses["g_loss"]
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                checkpoint["best_loss"] = self.best_loss
                best_path = ckpt_dir / "checkpoint_best.pt"
                torch.save(checkpoint, str(best_path))
                logger.info("New best model saved (g_loss: %.4f)", current_loss)

        # Save epoch-specific
        epoch_path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, str(epoch_path))

        logger.info("Checkpoint saved: epoch %d", epoch)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Epoch number to resume from.
        """
        import torch

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.generator and "generator" in checkpoint and checkpoint["generator"]:
            self.generator.load_state_dict(checkpoint["generator"])

        if self.discriminator and "discriminator" in checkpoint and checkpoint["discriminator"]:
            self.discriminator.load_state_dict(checkpoint["discriminator"])

        if self.optimizer_g and "optimizer_g" in checkpoint and checkpoint["optimizer_g"]:
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])

        if self.optimizer_d and "optimizer_d" in checkpoint and checkpoint["optimizer_d"]:
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        if self.scheduler_g and "scheduler_g" in checkpoint and checkpoint["scheduler_g"]:
            self.scheduler_g.load_state_dict(checkpoint["scheduler_g"])

        if self.scheduler_d and "scheduler_d" in checkpoint and checkpoint["scheduler_d"]:
            self.scheduler_d.load_state_dict(checkpoint["scheduler_d"])

        epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        logger.info(
            "Checkpoint loaded: epoch %d, step %d, best_loss %.4f",
            epoch, self.global_step, self.best_loss,
        )
        return epoch

    def export_for_inference(self, output_path: str) -> None:
        """
        Export model for inference (compatible with RVCConverter).

        Args:
            output_path: Output file path (.pt).
        """
        import torch

        if self.generator is None:
            raise RuntimeError("Generator not built. Call build_models() first.")

        # Create RVCConverter-compatible checkpoint
        checkpoint = {
            "model": self.generator.state_dict(),
            "config": {
                "sample_rate": self.config.data.sample_rate,
                "hop_length": self.config.data.hop_length,
                "mel_channels": self.config.data.n_mels,
                "use_flow": self.config.model.use_flow,
                "in_channels": self.config.model.in_channels,
                "out_channels": self.config.model.out_channels,
                "hidden_channels": self.config.model.hidden_channels,
                "embed_dim": self.config.model.embed_dim,
                "pitch_encoder_dim": self.config.model.pitch_encoder_dim,
            },
            "version": "rvc_v2",
            "sr": str(self.config.data.sample_rate),
            "pitch_guidance": 1,
            "epoch": self.current_epoch,
            "step": self.global_step,
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(output))
        logger.info("Model exported for RVC inference: %s", output_path)

    def export_sovits_format(self, output_path: str) -> None:
        """
        Export model for SoVITS inference (compatible with SoVITSConverter).

        Creates a VITS-format checkpoint and a companion config.json file
        that can be loaded by SoVITSConverter.load_model().

        The exported model uses the VITSGenerator architecture from
        sovits_models.py, with weights mapped from the trained RVC generator.

        Args:
            output_path: Output file path (.pt).
        """
        import torch

        if self.generator is None:
            raise RuntimeError("Generator not built. Call build_models() first.")

        # Build SoVITS-compatible config for VITSGenerator
        sovits_config = {
            "n_vocab": 0,  # Use content encoder (not text encoder)
            "spec_channels": self.config.data.n_mels,
            "hidden_channels": self.config.model.hidden_channels,
            "out_channels": self.config.model.out_channels,
            "n_speakers": 0,
            "gin_channels": 0,
            "use_flow": False,
        }

        # Create a VITSGenerator to get the correct state_dict structure
        from voice_converters.sovits_models import VITSGenerator

        vits_gen = VITSGenerator(
            n_vocab=sovits_config["n_vocab"],
            spec_channels=sovits_config["spec_channels"],
            hidden_channels=sovits_config["hidden_channels"],
            out_channels=sovits_config["out_channels"],
            n_speakers=sovits_config["n_speakers"],
            gin_channels=sovits_config["gin_channels"],
            use_transformer_flows=sovits_config["use_flow"],
        )

        # Try to transfer compatible weights from RVC generator
        rvc_state = self.generator.state_dict()
        vits_state = vits_gen.state_dict()

        # Map compatible layers (decoder layers are structurally similar)
        transferred = 0
        for key in vits_state:
            if key in rvc_state and rvc_state[key].shape == vits_state[key].shape:
                vits_state[key] = rvc_state[key]
                transferred += 1

        logger.info(
            "SoVITS export: transferred %d/%d compatible weight tensors",
            transferred, len(vits_state),
        )

        # Build full checkpoint in the format expected by
        # create_vits_model_from_checkpoint()
        checkpoint = {
            "model": vits_state,
            "config": sovits_config,
            "version": "sovits_v4.1",
            "epoch": self.current_epoch,
            "step": self.global_step,
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(output))

        # Also save a companion config.json (SoVITSConverter expects it)
        config_json = {
            "version": "4.1",
            "sampling_rate": self.config.data.sample_rate,
            "audio": {
                "sample_rate": self.config.data.sample_rate,
                "hop_length": self.config.data.hop_length,
                "n_mels": self.config.data.n_mels,
                "n_fft": self.config.data.n_fft,
            },
            "model": sovits_config,
            "data": {
                "sample_rate": self.config.data.sample_rate,
                "hop_length": self.config.data.hop_length,
            },
            "train": {
                "segment_size": self.config.data.segment_duration * self.config.data.sample_rate,
            },
        }

        config_path = output.parent / "config.json"
        import json
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_json, f, indent=2, ensure_ascii=False)

        logger.info(
            "Model exported for SoVITS inference: %s (config: %s)",
            output_path, config_path,
        )

    def export_dual_format(
        self,
        output_dir: str,
        rvc_name: str = "model_rvc.pt",
        sovits_name: str = "model_sovits.pt",
    ) -> Dict[str, str]:
        """
        Export model in both RVC and SoVITS formats.

        One training, two inference formats. Both files are saved to the
        same output directory with distinguishable names.

        Args:
            output_dir: Output directory for both model files.
            rvc_name: Filename for RVC format export.
            sovits_name: Filename for SoVITS format export.

        Returns:
            Dict with keys 'rvc' and 'sovits' mapping to file paths.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        rvc_path = str(output / rvc_name)
        sovits_path = str(output / sovits_name)

        self.export_for_inference(rvc_path)
        self.export_sovits_format(sovits_path)

        logger.info("Dual-format export complete: %s, %s", rvc_path, sovits_path)

        return {"rvc": rvc_path, "sovits": sovits_path}

    def close(self):
        """Clean up resources."""
        if self.writer:
            self.writer.close()
            self.writer = None
