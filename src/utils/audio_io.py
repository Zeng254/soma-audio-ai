"""
Audio I/O - Audio read/write tools

Provides unified audio file read/write interface.

Audio format conventions:
- Internal storage format: (channels, samples) - 2D array
- File read/write format: (samples, channels) - soundfile Standard format
- NumPy array: (samples,) - mono
"""

from pathlib import Path
from typing import Tuple, Optional, Union, List
import numpy as np


class AudioLoader:
    """
    Audio loader

    Supports multiple audio format loading and preprocessing.
    """

    SUPPORTED_FORMATS = {
        "wav", "mp3", "flac", "ogg", "aac",
        "m4a", "wma", "aiff", "amr", "opus"
    }

    # Common sample rates, used to check if channel count
    COMMON_SAMPLE_RATES = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000}

    def __init__(
        self,
        target_sr: Optional[int] = None,
        mono: bool = False,
        channel_first: Optional[bool] = None
    ):
        """
        Initialize loader

        Args:
            target_sr: Target sample rate, None means keep original
            mono: Whether to convert to mono
            channel_first: Explicitly specify input/output format
                          True: Returns (channels, samples)
                          False: Returns (samples, channels)
                          None: Automatic detection (default)
        """
        self.target_sr = target_sr
        self.mono = mono
        self.channel_first = channel_first

    def load(
        self,
        file_path: str,
        force_channel_first: Optional[bool] = None
    ) -> Tuple[np.ndarray, int]:
        """
        LoadAudioFile

        Args:
            file_path: AudioFile path
            force_channel_first: Force specify output format

        Returns:
            (audio_data, sample_rate)
            - channel_first=True: (channels, samples)
            - channel_first=False: (samples, channels)
        """
        path = Path(file_path)
        suffix = path.suffix[1:].lower()

        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}")

        # Determine output format
        output_channel_first = force_channel_first if force_channel_first is not None else self.channel_first
        if output_channel_first is None:
            output_channel_first = True  # Internal default uses channel_first

        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype='float32')
            # soundfile returns (samples, channels)

        except ImportError:
            # Fallback to pydub
            from pydub import AudioSegment
            audio_segment = AudioSegment.from_file(str(path))
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            sr = audio_segment.frame_rate

            # pydub returns (samples,), needs reshape
            if audio_segment.channels == 2:
                audio = audio.reshape((-1, 2))
            else:
                audio = audio.reshape(-1, 1)

        # Detect and convert channel format
        audio = self._ensure_channel_first(audio, sr)

        # Convert to mono
        if self.mono and audio.shape[0] > 1:
            audio = np.mean(audio, axis=0, keepdims=True)

        # Resampling
        if self.target_sr and self.target_sr != sr:
            audio = self._resample(audio, sr, self.target_sr)
            sr = self.target_sr

        # Decide whether to transpose based on output_channel_first
        if not output_channel_first:
            audio = audio.T

        return audio, sr

    def _ensure_channel_first(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Ensure audio format is (channels, samples)

        Detect current format through multiple methods:
        1. If explicit channel_first setting exists, use it
        2. If 1D array, directly add dimension
        3. Infer format by dimension size

        Args:
            audio: Input audio data
            sample_rate: Sample rate

        Returns:
            (channels, samples) FormatAudio
        """
        # If explicit setting exists
        if self.channel_first is not None:
            if self.channel_first and audio.ndim == 2 and audio.shape[0] > audio.shape[-1]:
                return audio.T
            elif not self.channel_first and audio.ndim == 2 and audio.shape[0] < audio.shape[-1]:
                return audio.T
            return audio

        # 1D array (mono)
        if audio.ndim == 1:
            return audio[np.newaxis, :]

        # 2D array needs inference
        if audio.ndim == 2:
            dim0, dim1 = audio.shape

            # If first dimension is common sample rate, second dimension is reasonable channel count
            if dim0 in self.COMMON_SAMPLE_RATES and dim1 <= 8:
                # This case is likely reversed
                if dim0 % sample_rate == 0 or dim0 > sample_rate:
                    return audio.T

            # If second dimension is common sample rate, first dimension is reasonable channel count
            if dim1 in self.COMMON_SAMPLE_RATES and dim0 <= 8:
                return audio

            # If first dimension is much larger than second, likely already channel_first
            if dim0 > dim1 * 2:
                return audio

            # If second dimension is much larger than first, likely (samples, channels)
            if dim1 > dim0 * 2:
                return audio.T

        # Conservative strategy: return original array by default (assume already channel_first)
        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resampling"""
        try:
            import librosa
            if audio.ndim == 2:
                result = np.zeros((audio.shape[0], int(audio.shape[1] * target_sr / orig_sr)))
                for ch in range(audio.shape[0]):
                    result[ch] = librosa.resample(
                        audio[ch],
                        orig_sr=orig_sr,
                        target_sr=target_sr
                    )
                return result
            else:
                return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            from scipy import signal
            if audio.ndim == 2:
                num_samples = int(audio.shape[1] * target_sr / orig_sr)
                result = np.zeros((audio.shape[0], num_samples))
                for ch in range(audio.shape[0]):
                    result[ch] = signal.resample(audio[ch], num_samples)
                return result
            else:
                num_samples = int(len(audio) * target_sr / orig_sr)
                return signal.resample(audio, num_samples)

    def load_segment(
        self,
        file_path: str,
        start: float,
        end: Optional[float] = None,
        force_channel_first: Optional[bool] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio segment

        Args:
            file_path: AudioFile path
            start: Start time (seconds)
            end: EndTime(seconds)
            force_channel_first: Force specify output format

        Returns:
            (audio_data, sample_rate)
        """
        audio, sr = self.load(file_path, force_channel_first=True)

        start_sample = int(start * sr)
        if end is not None:
            end_sample = int(end * sr)
            audio = audio[:, start_sample:end_sample]
        else:
            audio = audio[:, start_sample:]

        return audio, sr


class AudioSaver:
    """
    Audio saver

    Supports multiple audio format saving.
    """

    def __init__(
        self,
        normalize: bool = False,
        target_db: float = -3.0
    ):
        """
        Initialize saver

        Args:
            normalize: Whether to normalize
            target_db: Target decibel value
        """
        self.normalize = normalize
        self.target_db = target_db

    def save(
        self,
        audio: np.ndarray,
        file_path: str,
        sample_rate: int = 44100,
        format: Optional[str] = None,
        bit_depth: Optional[int] = 16,
        force_channel_first: Optional[bool] = None
    ) -> bool:
        """
        SaveAudioFile

        Args:
            audio: Audio data
            file_path: SavePath
            sample_rate: Sample rate
            format: AudioFormat
            bit_depth: Bit depth
            force_channel_first: Specify input format

        Returns:
            bool: Whether successful
        """
        # Prepare save path
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = path.suffix[1:].lower()

        # Prepare audio data (convert to soundfile required (samples, channels) format)
        audio = self._prepare_audio(audio, force_channel_first)

        try:
            import soundfile as sf

            # Determine subtype
            subtype = self._get_subtype(format, bit_depth)

            # WriteFile
            sf.write(
                str(path),
                audio,
                sample_rate,
                format=format.upper(),
                subtype=subtype
            )

            return True

        except ImportError:
            # Fallback to pydub
            return self._save_with_pydub(audio, path, sample_rate, format)

    def _prepare_audio(
        self,
        audio: np.ndarray,
        force_channel_first: Optional[bool] = None
    ) -> np.ndarray:
        """
        Prepare audio data, convert to (samples, channels) format

        Args:
            audio: Input audio
            force_channel_first: Specify input format

        Returns:
            (samples, channels) FormatAudio
        """
        # Ensure it is 2D array
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Determine whether transpose is needed
        needs_transpose = False

        if force_channel_first is not None:
            # Explicitly specified
            needs_transpose = force_channel_first
        elif self._looks_like_channel_first(audio):
            # Detect whether it looks like channel_first
            needs_transpose = True

        if needs_transpose and audio.shape[0] < audio.shape[1]:
            audio = audio.T

        # Normalization
        if self.normalize:
            audio = self._normalize(audio)

        # Limit range
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def _looks_like_channel_first(self, audio: np.ndarray) -> bool:
        """
        Detect whether audio might be channel_first format

        Detectionmethod：
        1. If first dimension <= 8, likely channel count
        2. If second dimension is sample rate multiple, likely sample count
        """
        if audio.ndim != 2:
            return False

        channels, samples = audio.shape

        # Channel count usually <= 8
        if channels > 8:
            return False

        # If first dimension is common sample rate, dimensions may be swapped
        COMMON_SAMPLE_RATES = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000}
        if channels in COMMON_SAMPLE_RATES and samples <= 8:
            return True

        return False

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """NormalizationAudio"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            target_linear = 10 ** (self.target_db / 20)
            audio = audio * (target_linear / max_val)
        return audio

    def _get_subtype(self, format: str, bit_depth: int) -> str:
        """Get audio subclass type"""
        lossless = {"wav", "flac", "aiff"}
        lossy = {"mp3", "ogg", "aac", "m4a"}

        if format.lower() in lossless:
            bit_map = {
                16: "PCM_16",
                24: "PCM_24",
                32: "PCM_32",
            }
        elif format.lower() in lossy:
            return "VORBIS" if format.lower() == "ogg" else "AAC"
        else:
            return "PCM_16"

        return bit_map.get(bit_depth, "PCM_16")

    def _save_with_pydub(
        self,
        audio: np.ndarray,
        path: Path,
        sample_rate: int,
        format: str
    ) -> bool:
        """Uses pydub Save"""
        from pydub import AudioSegment

        # Ensure format is (samples, channels)
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T

        channels = audio.shape[1] if audio.ndim > 1 else 1

        # Convert back to int16
        audio_int = (audio * 32767).astype(np.int16)

        # Create AudioSegment
        segment = AudioSegment(
            audio_int.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=channels,
        )

        segment.export(str(path), format=format)
        return True

    def save_tracks(
        self,
        tracks: dict,
        output_dir: str,
        sample_rate: int = 44100,
        format: str = "wav",
    ) -> dict:
        """
        Save multiple audio tracks

        Args:
            tracks: track dictionary {name: audio}
            output_dir: output directory
            sample_rate: Sample rate
            format: AudioFormat

        Returns:
            dict: save result {name: path}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        for name, audio in tracks.items():
            file_path = output_path / f"{name}.{format}"
            self.save(audio, str(file_path), sample_rate, format)
            saved_paths[name] = str(file_path)

        return saved_paths
