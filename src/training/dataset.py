"""
RVC Training Dataset Module

Provides PyTorch Dataset and DataLoader for RVC model training:
- Custom Dataset loading preprocessed numpy segments
- Auto-resampling to target sample rate
- Fixed-length padding/truncation
- Multi-process DataLoader support
- Train/validation split
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config import DataConfig

logger = logging.getLogger(__name__)

# Try to import torch Dataset for inheritance
try:
    from torch.utils.data import Dataset as TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Create a dummy base class if torch is not available
    class TorchDataset:
        """Dummy base class when torch is not available."""
        pass


class RVCDataset(TorchDataset):
    """
    PyTorch-compatible Dataset for RVC training.

    Loads preprocessed numpy segments from a directory and provides
    fixed-length tensors for training.

    Inherits from torch.utils.data.Dataset when PyTorch is available,
    making it fully compatible with torch.utils.data.DataLoader.
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[DataConfig] = None,
        segment_length: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing preprocessed .npy segment files.
            config: Data configuration. Uses defaults if None.
            segment_length: Fixed segment length in samples. If None,
                computed from config.segment_duration * config.sample_rate.
        """
        self.config = config or DataConfig()
        self.data_dir = Path(data_dir)

        if segment_length is None:
            self.segment_length = int(
                self.config.segment_duration * self.config.sample_rate
            )
        else:
            self.segment_length = segment_length

        # Collect all .npy files
        self.file_paths: List[Path] = sorted(self.data_dir.glob("*.npy"))

        if not self.file_paths:
            logger.warning("No .npy files found in %s", data_dir)

        logger.info(
            "RVCDataset initialized: %d segments from %s (segment_length=%d)",
            len(self.file_paths), data_dir, self.segment_length,
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a training sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys:
                - "audio": np.ndarray of shape [segment_length]
                - "mel": np.ndarray of shape [n_mels, time]
        """
        if idx >= len(self.file_paths):
            raise IndexError(f"Index {idx} out of range (dataset size: {len(self)})")

        # Load segment
        audio = np.load(str(self.file_paths[idx]))

        # Pad or truncate to fixed length
        audio = self._fix_length(audio)

        # Compute mel spectrogram
        mel = self._compute_mel(audio)

        return {
            "audio": audio,
            "mel": mel,
        }

    def _fix_length(self, audio: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to fixed segment length."""
        if len(audio) >= self.segment_length:
            return audio[: self.segment_length].astype(np.float32)
        else:
            padded = np.zeros(self.segment_length, dtype=np.float32)
            padded[: len(audio)] = audio
            return padded

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from audio.

        Uses simple STFT-based mel computation to avoid heavy dependencies.
        """
        try:
            import librosa
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            return mel_db.astype(np.float32)
        except ImportError:
            pass

        # Fallback: simple STFT-based mel
        return self._simple_mel(audio)

    def _simple_mel(self, audio: np.ndarray) -> np.ndarray:
        """Simple mel spectrogram computation without librosa."""
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length
        n_mels = self.config.n_mels
        sr = self.config.sample_rate

        # Windowed STFT
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        if n_frames <= 0:
            return np.zeros((n_mels, 1), dtype=np.float32)

        window = np.hanning(n_fft).astype(np.float32)
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            stft[:, i] = spectrum

        # Power spectrum
        power = np.abs(stft) ** 2

        # Simple mel filterbank (triangular filters)
        mel_fb = self._mel_filterbank(sr, n_fft, n_mels)
        mel_spec = mel_fb @ power

        # Log scale
        mel_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
        return mel_db.astype(np.float32)

    def _mel_filterbank(
        self, sr: int, n_fft: int, n_mels: int
    ) -> np.ndarray:
        """Create mel filterbank matrix."""
        fmin = 0.0
        fmax = sr / 2.0

        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        bin_points = np.clip(bin_points, 0, n_fft // 2)

        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                if center > left:
                    fb[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    fb[i, j] = (right - j) / (right - center)

        return fb


def create_dataloader(
    dataset: RVCDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    """
    Create a PyTorch DataLoader from an RVCDataset.

    Args:
        dataset: RVCDataset instance.
        batch_size: Batch size.
        shuffle: Shuffle data if True.
        num_workers: Number of worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop last incomplete batch.

    Returns:
        PyTorch DataLoader.
    """
    try:
        import torch
        from torch.utils.data import DataLoader as TorchDataLoader

        def collate_fn(batch):
            """Custom collate to handle dict-based samples."""
            audios = np.stack([b["audio"] for b in batch])
            mels = np.stack([b["mel"] for b in batch])
            return {
                "audio": torch.from_numpy(audios),
                "mel": torch.from_numpy(mels),
            }

        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
    except ImportError:
        raise ImportError(
            "PyTorch is required for DataLoader. Install with: uv add torch"
        )


def split_dataset(
    data_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Split preprocessed data into train and validation sets.

    Args:
        data_dir: Directory containing .npy files.
        train_ratio: Ratio of training data (0, 1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_files, val_files).
    """
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob("*.npy"))

    if not all_files:
        return [], []

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_files))
    split_idx = int(len(all_files) * train_ratio)

    train_files = [all_files[i] for i in indices[:split_idx]]
    val_files = [all_files[i] for i in indices[split_idx:]]

    logger.info(
        "Dataset split: %d train, %d val (ratio=%.2f)",
        len(train_files), len(val_files), train_ratio,
    )
    return train_files, val_files
