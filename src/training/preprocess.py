"""
RVC Audio Preprocessing Module

Provides audio preprocessing pipeline for RVC model training:
- Multi-format audio loading (wav/mp3/flac/ogg/m4a)
- Silence trimming (energy-based threshold)
- Fixed-length segmentation (with overlap support)
- Volume normalization (LUFS / peak)
- Data quality validation (sample rate, duration, SNR)
- Output standardized numpy array segments
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .config import DataConfig

logger = logging.getLogger(__name__)


def load_audio(
    file_path: str,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with optional resampling.

    Args:
        file_path: Path to audio file.
        target_sr: Target sample rate. If None, keep original.
        mono: Convert to mono if True.

    Returns:
        Tuple of (audio_data, sample_rate).

    Raises:
        FileNotFoundError: If audio file does not exist.
        ValueError: If audio format is unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aiff"}
    if path.suffix.lower() not in supported:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. Supported: {supported}"
        )

    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        return audio, sr
    except ImportError:
        logger.warning("librosa not available, falling back to soundfile")

    try:
        import soundfile as sf
        audio, sr = sf.read(file_path, always_2d=False)
        if mono and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if target_sr is not None and target_sr != sr:
            audio = _resample(audio, sr, target_sr)
            sr = target_sr
        return audio.astype(np.float32), sr
    except ImportError:
        raise ImportError(
            "Neither librosa nor soundfile is installed. "
            "Install one with: uv add librosa  OR  uv add soundfile"
        )


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS energy of audio signal."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_snr(audio: np.ndarray, noise_floor: Optional[float] = None) -> float:
    """
    Estimate SNR in dB.

    Args:
        audio: Audio signal.
        noise_floor: Noise floor RMS. If None, estimated from quietest 10% of frames.

    Returns:
        Estimated SNR in dB.
    """
    if len(audio) == 0:
        return 0.0

    frame_size = max(256, len(audio) // 100)
    n_frames = len(audio) // frame_size
    if n_frames < 2:
        return 20.0  # default for very short audio

    frame_energies = []
    for i in range(n_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        frame_energies.append(compute_rms(frame))

    frame_energies = np.array(frame_energies)
    if noise_floor is None:
        noise_floor = float(np.percentile(frame_energies, 10))

    signal_rms = float(np.max(frame_energies))
    if noise_floor < 1e-10:
        return 60.0  # very clean signal

    snr = 20.0 * np.log10(signal_rms / max(noise_floor, 1e-10))
    return float(snr)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.3,
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.

    Args:
        audio: Audio signal.
        sample_rate: Sample rate.
        threshold_db: Silence threshold in dB.
        min_silence_duration: Minimum silence duration to trim (seconds).

    Returns:
        Trimmed audio.
    """
    if len(audio) == 0:
        return audio

    threshold = 10.0 ** (threshold_db / 20.0)
    min_silence_samples = int(min_silence_duration * sample_rate)

    # Compute frame energy
    frame_size = max(256, sample_rate // 50)  # ~20ms frames
    hop = frame_size // 2
    n_frames = (len(audio) - frame_size) // hop + 1

    if n_frames <= 0:
        return audio

    energies = np.array([
        compute_rms(audio[i * hop : i * hop + frame_size])
        for i in range(n_frames)
    ])

    # Find voiced region
    voiced = energies > threshold
    if not np.any(voiced):
        return audio

    first_voiced = np.argmax(voiced)
    last_voiced = len(voiced) - np.argmax(voiced[::-1]) - 1

    # Convert frame indices to sample indices
    start_sample = max(0, first_voiced * hop - min_silence_samples)
    end_sample = min(len(audio), (last_voiced + 1) * hop + min_silence_samples)

    return audio[start_sample:end_sample]


def normalize_peak(
    audio: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    """
    Normalize audio to target peak amplitude.

    Args:
        audio: Audio signal.
        target_peak: Target peak amplitude (0, 1].

    Returns:
        Normalized audio.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return audio
    return audio * (target_peak / peak)


def normalize_lufs(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -23.0,
) -> np.ndarray:
    """
    Normalize audio to target LUFS (approximate).

    Uses a simplified loudness estimation based on RMS energy
    with A-weighting approximation.

    Args:
        audio: Audio signal.
        sample_rate: Sample rate.
        target_lufs: Target loudness in LUFS.

    Returns:
        Normalized audio.
    """
    if len(audio) == 0:
        return audio

    # Approximate loudness using RMS (simplified LUFS)
    rms = compute_rms(audio)
    if rms < 1e-10:
        return audio

    # Approximate LUFS from RMS (simplified model)
    current_lufs = 20.0 * np.log10(rms) - 0.691  # rough approximation
    gain_db = target_lufs - current_lufs
    gain = 10.0 ** (gain_db / 20.0)

    normalized = audio * gain
    # Prevent clipping
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized * (0.99 / peak)

    return normalized


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    segment_duration: float = 3.0,
    overlap: float = 0.1,
) -> List[np.ndarray]:
    """
    Split audio into fixed-length segments with overlap.

    Args:
        audio: Audio signal.
        sample_rate: Sample rate.
        segment_duration: Duration of each segment in seconds.
        overlap: Overlap ratio between segments [0, 1).

    Returns:
        List of audio segments.
    """
    if len(audio) == 0:
        return []

    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(segment_samples * (1.0 - overlap))

    if hop_samples <= 0:
        hop_samples = 1

    segments = []
    start = 0
    while start + segment_samples <= len(audio):
        segments.append(audio[start : start + segment_samples].copy())
        start += hop_samples

    # Handle remaining audio (pad if needed)
    if start < len(audio) and len(audio) - start > segment_samples // 2:
        remaining = audio[start:]
        if len(remaining) < segment_samples:
            padded = np.zeros(segment_samples, dtype=np.float32)
            padded[: len(remaining)] = remaining
            segments.append(padded)
        else:
            segments.append(remaining[:segment_samples].copy())

    return segments


def validate_audio_quality(
    audio: np.ndarray,
    sample_rate: int,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    min_snr: float = 10.0,
    expected_sr: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate audio quality for training.

    Args:
        audio: Audio signal.
        sample_rate: Sample rate.
        min_duration: Minimum duration in seconds.
        max_duration: Maximum duration in seconds.
        min_snr: Minimum SNR in dB.
        expected_sr: Expected sample rate. If None, skip check.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []

    if len(audio) == 0:
        issues.append("Audio is empty")
        return False, issues

    duration = len(audio) / sample_rate
    if duration < min_duration:
        issues.append(
            f"Audio too short: {duration:.2f}s < {min_duration:.2f}s"
        )
    if duration > max_duration:
        issues.append(
            f"Audio too long: {duration:.2f}s > {max_duration:.2f}s"
        )

    if expected_sr is not None and sample_rate != expected_sr:
        issues.append(
            f"Sample rate mismatch: {sample_rate} != {expected_sr}"
        )

    snr = compute_snr(audio)
    if snr < min_snr:
        issues.append(f"Low SNR: {snr:.1f} dB < {min_snr:.1f} dB")

    # Check for clipping
    if np.max(np.abs(audio)) > 0.999:
        issues.append("Audio is clipping (peak > 0.999)")

    # Check for silence
    rms = compute_rms(audio)
    if rms < 1e-6:
        issues.append("Audio appears to be silent")

    return len(issues) == 0, issues


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for RVC training.

    Applies the full preprocessing chain:
    1. Load audio
    2. Trim silence
    3. Normalize volume
    4. Validate quality
    5. Segment into fixed-length clips
    6. Save as numpy arrays
    """

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize preprocessor.

        Args:
            config: Data configuration. Uses defaults if None.
        """
        self.config = config or DataConfig()

    def process_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Process a single audio file through the full pipeline.

        Args:
            file_path: Path to audio file.
            output_dir: If provided, save segments to this directory.

        Returns:
            List of processed audio segments.
        """
        logger.info("Processing: %s", file_path)

        # 1. Load audio
        audio, sr = load_audio(file_path, target_sr=self.config.sample_rate)

        # 2. Trim silence
        audio = trim_silence(
            audio, sr,
            threshold_db=self.config.silence_threshold,
        )

        if len(audio) == 0:
            logger.warning("File %s is empty after silence trimming", file_path)
            return []

        # 3. Normalize
        if self.config.normalize_mode == "lufs":
            audio = normalize_lufs(audio, sr, target_lufs=self.config.target_lufs)
        else:
            audio = normalize_peak(audio, target_peak=self.config.target_peak)

        # 4. Validate quality
        is_valid, issues = validate_audio_quality(
            audio, sr,
            min_duration=self.config.min_clip_duration,
            max_duration=self.config.max_clip_duration,
        )
        if not is_valid:
            logger.warning(
                "Quality issues in %s: %s", file_path, issues
            )
            if any("empty" in i.lower() or "silent" in i.lower() for i in issues):
                return []

        # 5. Segment
        segments = segment_audio(
            audio, sr,
            segment_duration=self.config.segment_duration,
            overlap=self.config.overlap,
        )

        # 6. Save if output_dir provided
        if output_dir and segments:
            self._save_segments(segments, file_path, output_dir)

        logger.info(
            "Processed %s: %d segments (%.1fs each)",
            file_path, len(segments), self.config.segment_duration,
        )
        return segments

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = True,
    ) -> dict:
        """
        Process all audio files in a directory.

        Args:
            input_dir: Input directory containing audio files.
            output_dir: Output directory for processed segments.
            recursive: Search subdirectories if True.

        Returns:
            Dict with processing statistics.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect audio files
        extensions = [f"*.{ext}" for ext in self.config.supported_formats]
        audio_files = []
        for ext in extensions:
            if recursive:
                audio_files.extend(input_path.rglob(ext))
            else:
                audio_files.extend(input_path.glob(ext))

        audio_files = sorted(set(audio_files))
        logger.info("Found %d audio files in %s", len(audio_files), input_dir)

        stats = {
            "total_files": len(audio_files),
            "processed_files": 0,
            "failed_files": 0,
            "total_segments": 0,
        }

        for audio_file in audio_files:
            try:
                segments = self.process_file(str(audio_file), output_dir)
                if segments:
                    stats["processed_files"] += 1
                    stats["total_segments"] += len(segments)
                else:
                    stats["failed_files"] += 1
            except Exception as e:
                logger.error("Failed to process %s: %s", audio_file, e)
                stats["failed_files"] += 1

        logger.info(
            "Preprocessing complete: %d/%d files, %d segments",
            stats["processed_files"],
            stats["total_files"],
            stats["total_segments"],
        )
        return stats

    def _save_segments(
        self,
        segments: List[np.ndarray],
        source_file: str,
        output_dir: str,
    ) -> None:
        """Save segments as numpy files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stem = Path(source_file).stem
        for i, segment in enumerate(segments):
            save_path = output_path / f"{stem}_seg{i:04d}.npy"
            np.save(str(save_path), segment)
