"""
Demucs Separator - Based on Demucs Audio separator

Supports vocals/accompaniment/drums/bass and other 4-track separation.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from src.separators.base import BaseSeparator, SeparationResult
from src.utils.audio_io import AudioLoader


class DemucsSeparator(BaseSeparator):
    """
    Demucs Audio separator

    Based on Facebook Research  Demucs deep learning model，
    Supports high-quality audio source separation.

    Supported Models:
    - hdemucs.mmi: General model, supports 4-track separation
    - htdemucs: General model, supports 4-track separation
    - htdemucs_ft: Fine-tuned model, higher quality but slower
    - htdemucs_mmi: MMI version

    Supported tracks:
    - vocals: Vocals
    - drums: Drums
    - bass: Bass
    - other: Other instruments
    """

    # Demucs model standard track order
    DEFAULT_TRACKS = ["vocals", "drums", "bass", "other"]

    def __init__(
        self,
        model_name: str = "htdemucs",
        sample_rate: int = 44100,
        device: Optional[str] = None,
        progress: bool = True,
    ):
        """
        Initialize Demucs Separator

        Args:
            model_name: ModelName
            sample_rate: Sample rate
            device: Run device
            progress: Whether to show progress
        """
        super().__init__(sample_rate, device)
        self.model_name = model_name
        self.progress = progress
        self._model = None
        self._demucs = None

    def _load_model(self):
        """DelayLoad Demucs Model"""
        if self._model is None:
            try:
                from demucs import pretrained
                from demucs.pretrained import get_model
                # Demucs 4.0+ API: get_model returns a model object
                model_bundle = get_model(self.model_name)
                # Get actual model from bundle
                self._model = model_bundle
                self._demucs = model_bundle.get_model()
                if self.device != "cpu":
                    self._demucs.to(self.device)
                self._demucs.eval()
            except ImportError:
                raise ImportError(
                    "Demucs not installed. Please run: uv add demucs"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Demucs model: {e}")

    def get_model_name(self) -> str:
        return f"Demucs-{self.model_name}"

    def get_available_tracks(self) -> list:
        """Get available track list"""
        if self._demucs is not None:
            return list(self._demucs.sources)
        return list(self.DEFAULT_TRACKS)

    def separate(self, audio_path: str, **kwargs) -> SeparationResult:
        """
        FromFile pathSeparationAudio

        Args:
            audio_path: Input audio file path
            **kwargs: OtherParameter (e.g. output_dir)

        Returns:
            SeparationResult: Separation result
        """
        loader = AudioLoader(channel_first=True)
        audio, sr = loader.load(audio_path, force_channel_first=True)

        return self.separate_array(audio, sr, **kwargs)

    def separate_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        **kwargs
    ) -> SeparationResult:
        """
        Perform separation on audio array

        Args:
            audio: Audio data, format is (channels, samples) or (samples,)
            sample_rate: Sample rate
            **kwargs: OtherParameter

        Returns:
            SeparationResult: Separation result
        """
        self._load_model()

        # Validate and normalize input
        audio = self.validate_audio_input(audio)

        # Resample to model required sample rate
        if sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)

        # Convert to model input format
        audio_tensor = self._prepare_tensor(audio)

        # ExecuteSeparation - Demucs 4.0+ API
        import torch
        with torch.no_grad():
            sources = self._demucs(audio_tensor)

        # Extract each track to result object
        result = self._extract_tracks(sources)

        return result

    def _extract_tracks(self, sources: "torch.Tensor") -> SeparationResult:
        """
        Extract each track from model output

        Args:
            sources: Model output tensor, shape is (batch, tracks, samples)
                     Track order is defined by Demucs

        Returns:
            SeparationResult: Contains each track result
        """
        result = SeparationResult(sample_rate=self.sample_rate)

        # Get Demucs defined track order
        tracks = self._demucs.sources

        # Ensure output is on CPU and convert to numpy
        sources_np = sources[0].cpu().numpy()  # (tracks, samples)

        # Extract each track
        for i, track_name in enumerate(tracks):
            if i < len(sources_np):
                track_data = sources_np[i]

                # Assign directly to result object
                if track_name == "vocals":
                    result.vocals = track_data
                elif track_name == "drums":
                    result.drums = track_data
                elif track_name == "bass":
                    result.bass = track_data
                elif track_name == "other":
                    result.other = track_data

        return result

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """ResamplingAudio"""
        try:
            import librosa
            if audio.ndim == 2:
                # Convert to (samples, channels) for librosa resampling
                audio_t = audio.T
                result = np.zeros((int(audio_t.shape[0] * target_sr / orig_sr), audio_t.shape[1]))
                for ch in range(audio_t.shape[1]):
                    result[:, ch] = librosa.resample(
                        audio_t[:, ch],
                        orig_sr=orig_sr,
                        target_sr=target_sr
                    )
                return result.T  # Transpose back to (channels, samples)
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

    def _prepare_tensor(self, audio: np.ndarray) -> "torch.Tensor":
        """Prepare model input tensor"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required. Please run: uv add torch torchaudio")

        # Convert to tensor (batch, channels, samples)
        tensor = torch.from_numpy(audio).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        elif tensor.dim() == 2:
            if audio.shape[0] > audio.shape[1]:
                # (channels, samples) -> (1, channels, samples)
                tensor = tensor.unsqueeze(0)
            else:
                # (samples, channels) -> (1, channels, samples)
                tensor = tensor.T.unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(0)

        # Move to device
        tensor = tensor.to(self.device)

        return tensor
