"""
RVC Inference Pipeline

Complete inference pipeline for RVC (Retrieval-based Voice Conversion) models.
Reuses the training module's FeaturePipeline for consistent feature extraction.

Key components:
    - RVCInference: Main inference pipeline class
    - VocoderWrapper: Lazy-loading vocoder integration (NSF-HiFiGAN / HiFi-GAN)

Usage:
    inference = RVCInference(model_path="path/to/model.pth")
    output_audio = inference.convert(source_audio, sample_rate=16000)

    # With pitch adjustment
    output_audio = inference.convert(source_audio, transpose=12)  # +12 semitones

    # Batch inference
    outputs = inference.convert_batch([audio1, audio2, audio3])
"""

import logging
from typing import Optional, Tuple, Union, List

import numpy as np

logger = logging.getLogger(__name__)


class VocoderWrapper:
    """
    Lazy-loading vocoder wrapper.

    Supports NSF-HiFiGAN and standard HiFiGAN vocoders commonly used with RVC.
    Falls back to a simple linear interpolation upsampler if no vocoder is available.

    Attributes:
        vocoder: The loaded vocoder model (or None if using fallback).
        device: Device the vocoder is loaded on.
        using_fallback: Whether the vocoder is using the fallback upsampler.
    """

    def __init__(
        self,
        vocoder_path: Optional[str] = None,
        vocoder_type: str = "hifigan",
        device: str = "cpu",
        sample_rate: int = 40000,
    ):
        """
        Initialize the vocoder wrapper.

        Args:
            vocoder_path: Path to vocoder checkpoint. If None, uses fallback upsampler.
            vocoder_type: Type of vocoder ("hifigan", "nsf_hifigan").
            device: Device to load the vocoder on.
            sample_rate: Target sample rate for vocoder output.
        """
        self.vocoder_path = vocoder_path
        self.vocoder_type = vocoder_type
        self.device = device
        self.sample_rate = sample_rate
        self.vocoder = None
        self._is_loaded = False
        self._using_fallback = False

    @property
    def is_loaded(self) -> bool:
        """Whether the vocoder has been loaded."""
        return self._is_loaded

    @property
    def using_fallback(self) -> bool:
        """Whether the vocoder is using the fallback upsampler."""
        return self._using_fallback

    def load(self) -> None:
        """
        Load the vocoder model.

        If vocoder_path is None or loading fails, falls back to a simple
        linear interpolation upsampler.
        """
        if self._is_loaded:
            return

        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not available, using fallback vocoder")
            self._using_fallback = True
            self._is_loaded = True
            return

        if self.vocoder_path is None:
            logger.info("No vocoder path specified, using fallback upsampler")
            self._using_fallback = True
            self._is_loaded = True
            return

        try:
            self._load_vocoder_from_path()
        except Exception as e:
            logger.warning("Failed to load vocoder from %s: %s. Using fallback.", self.vocoder_path, e)
            self._using_fallback = True

        self._is_loaded = True

    def _load_vocoder_from_path(self) -> None:
        """Load vocoder from checkpoint file."""
        import torch

        checkpoint = torch.load(self.vocoder_path, map_location=self.device, weights_only=False)

        # Try to detect vocoder type from checkpoint
        if isinstance(checkpoint, dict):
            if "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Import HiFiGAN from voice_converters
        try:
            from voice_converters.rvc_models import RVCGenerator
            self.vocoder = RVCGenerator(
                in_channels=256,
                out_channels=1,
            )
            self.vocoder.load_state_dict(state_dict, strict=False)
            self.vocoder.to(self.device)
            self.vocoder.eval()
            logger.info("Loaded HiFiGAN vocoder from %s", self.vocoder_path)
        except Exception as e:
            logger.warning("Failed to load vocoder architecture: %s", e)
            self._using_fallback = True

    def synthesize(
        self,
        features: "np.ndarray",
        f0: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        """
        Synthesize audio from features using the vocoder.

        Args:
            features: Feature tensor, shape (feature_dim, num_frames) or (1, feature_dim, num_frames).
            f0: Optional F0 contour, shape (num_frames,).

        Returns:
            Synthesized audio waveform, shape (num_samples,).
        """
        import torch

        if not self._is_loaded:
            self.load()

        # Convert to torch tensor
        if isinstance(features, np.ndarray):
            features_t = torch.from_numpy(features).float().to(self.device)
        else:
            features_t = features.to(self.device)

        # Ensure 3D: (1, feature_dim, num_frames)
        if features_t.ndim == 2:
            features_t = features_t.unsqueeze(0)

        if self._using_fallback or self.vocoder is None:
            return self._fallback_upsample(features_t)

        # Run vocoder
        with torch.no_grad():
            if f0 is not None:
                if isinstance(f0, np.ndarray):
                    f0_t = torch.from_numpy(f0).float().to(self.device)
                else:
                    f0_t = f0.to(self.device)

                if f0_t.ndim == 1:
                    f0_t = f0_t.unsqueeze(0)

                # Match frame count
                target_frames = features_t.shape[-1]
                if f0_t.shape[-1] != target_frames:
                    f0_t = torch.nn.functional.interpolate(
                        f0_t.unsqueeze(1),
                        size=target_frames,
                        mode="linear",
                    ).squeeze(1)

                output = self.vocoder(features_t, f0_t.unsqueeze(1))
            else:
                # Some vocoders can work without F0
                output = self.vocoder(features_t)

        # Convert to numpy
        audio = output.squeeze().cpu().numpy()
        return audio

    def _fallback_upsample(self, features: "torch.Tensor") -> "np.ndarray":
        """
        Fallback: simple linear interpolation upsampler.

        Converts features to audio by upsampling from frame rate to sample rate.
        This produces low-quality output but allows the pipeline to work without
        a trained vocoder.
        """
        import torch

        # features shape: (1, feature_dim, num_frames)
        num_frames = features.shape[-1]
        # Upsample to target sample rate
        target_samples = int(num_frames * self.sample_rate / 100.0)  # ~100 fps

        # Average across feature dims to get mono signal
        mono = features.mean(dim=1)  # (1, num_frames)

        # Linear interpolation upsample
        upsampled = torch.nn.functional.interpolate(
            mono.unsqueeze(0),
            size=target_samples,
            mode="linear",
        ).squeeze()

        # Normalize
        if upsampled.abs().max() > 0:
            upsampled = upsampled / upsampled.abs().max() * 0.9

        return upsampled.cpu().numpy()


def _transpose_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    """
    Transpose F0 contour by a number of semitones.

    Args:
        f0: F0 contour in Hz, shape (num_frames,).
        semitones: Number of semitones to transpose (positive = up, negative = down).

    Returns:
        Transposed F0 contour in Hz.
    """
    # Convert Hz to semitones (relative to a reference), shift, convert back
    # Using the formula: f_new = f_old * 2^(semitones/12)
    factor = 2.0 ** (semitones / 12.0)
    return f0 * factor


class RVCInference:
    """
    Complete RVC inference pipeline.

    Loads a trained RVC model checkpoint and provides end-to-end voice conversion:
    1. Extract HuBERT features from source audio (using FeaturePipeline)
    2. Extract F0 contour from source audio
    3. Optionally transpose F0 for pitch adjustment
    4. Run RVC generator to produce converted features
    5. Run vocoder to synthesize output audio

    Attributes:
        model: The loaded RVC model.
        feature_pipeline: The feature extraction pipeline (HuBERT + F0).
        vocoder: The vocoder for audio synthesis.
        device: Device for inference.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocoder_path: Optional[str] = None,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        output_sample_rate: int = 40000,
        hubert_model: str = "hubert_base",
        f0_method: str = "dio",
        hop_length: int = 160,
    ):
        """
        Initialize the RVC inference pipeline.

        Args:
            model_path: Path to RVC model checkpoint. If None, model must be loaded later.
            vocoder_path: Path to vocoder checkpoint. If None, uses fallback upsampler.
            device: Device for inference ("cpu" or "cuda"). Auto-detected if None.
            sample_rate: Input audio sample rate (Hz).
            output_sample_rate: Output audio sample rate (Hz).
            hubert_model: HuBERT model name for feature extraction.
            f0_method: F0 extraction method ("dio", "harvest", "yin", "pyin").
            hop_length: Hop length for feature extraction.
        """
        import torch

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.sample_rate = sample_rate
        self.output_sample_rate = output_sample_rate
        self.hop_length = hop_length

        # Lazy-loaded components
        self._model_path = model_path
        self._vocoder_path = vocoder_path
        self._hubert_model = hubert_model
        self._f0_method = f0_method

        self.model = None
        self._model_loaded = False

        # Feature pipeline (lazy loaded)
        self._feature_pipeline = None

        # Vocoder (lazy loaded)
        self._vocoder = VocoderWrapper(
            vocoder_path=vocoder_path,
            device=self.device,
            sample_rate=output_sample_rate,
        )

        # Load model if path provided
        if model_path is not None:
            self.load_model()

    @property
    def is_model_loaded(self) -> bool:
        """Whether the RVC model has been loaded."""
        return self._model_loaded

    @property
    def feature_pipeline(self):
        """Get the feature pipeline, creating it if needed."""
        if self._feature_pipeline is None:
            from training.feature_extractor import FeaturePipeline
            self._feature_pipeline = FeaturePipeline(
                model_name=self._hubert_model,
                device=self.device,
                f0_method=self._f0_method,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        return self._feature_pipeline

    @property
    def vocoder(self) -> VocoderWrapper:
        """Get the vocoder wrapper."""
        return self._vocoder

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the RVC model from checkpoint.

        Args:
            model_path: Path to checkpoint. Uses the path from __init__ if None.
        """
        import torch

        path = model_path or self._model_path
        if path is None:
            raise ValueError("No model path specified")

        self._model_path = path

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}") from e

        # Build model from checkpoint
        from voice_converters.rvc_models import create_rvc_model_from_checkpoint

        self.model = create_rvc_model_from_checkpoint(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True

        logger.info("Loaded RVC model from %s on %s", path, self.device)

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        transpose: float = 0.0,
        f0_curve: Optional[np.ndarray] = None,
        return_features: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Convert source audio using the RVC model.

        Args:
            audio: Source audio waveform, shape (num_samples,) or (1, num_samples).
            sample_rate: Sample rate of input audio. Uses pipeline default if None.
            transpose: Pitch transpose in semitones (positive = up, negative = down).
            f0_curve: Optional custom F0 curve. If None, extracted from audio.
            return_features: If True, also return (features, f0) tuple.

        Returns:
            If return_features is False:
                output_audio: Converted audio waveform, shape (num_output_samples,).
            If return_features is True:
                (output_audio, features, f0) tuple.
        """
        import torch

        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        sr = sample_rate or self.sample_rate

        # Flatten audio to 1D
        if audio.ndim > 1:
            audio = audio.flatten()

        # Step 1: Extract features
        features, f0 = self.feature_pipeline.extract(audio, sr)

        # Step 2: Apply pitch transpose
        if transpose != 0.0:
            f0 = _transpose_f0(f0, transpose)

        # Step 3: Use custom F0 if provided
        if f0_curve is not None:
            f0 = f0_curve

        # Step 4: Run RVC model inference
        features_t = torch.from_numpy(features).float().to(self.device)
        f0_t = torch.from_numpy(f0).float().to(self.device)

        # Ensure 3D: (1, feature_dim, num_frames)
        if features_t.ndim == 2:
            features_t = features_t.unsqueeze(0)
        if f0_t.ndim == 1:
            f0_t = f0_t.unsqueeze(0)

        with torch.no_grad():
            output = self.model.inference(features_t, f0_t)

        # Step 5: Convert to numpy
        output_features = output.squeeze(0).cpu().numpy()

        # Step 6: Run vocoder to synthesize audio
        output_audio = self._vocoder.synthesize(output_features, f0)

        if return_features:
            return output_audio, features, f0
        return output_audio

    def convert_batch(
        self,
        audio_list: List[np.ndarray],
        sample_rate: Optional[int] = None,
        transpose: float = 0.0,
    ) -> List[np.ndarray]:
        """
        Convert a batch of audio files.

        Args:
            audio_list: List of audio waveforms.
            sample_rate: Sample rate of input audio.
            transpose: Pitch transpose in semitones.

        Returns:
            List of converted audio waveforms.
        """
        results = []
        for audio in audio_list:
            result = self.convert(audio, sample_rate=sample_rate, transpose=transpose)
            results.append(result)
        return results

    def save_output(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: Optional[int] = None,
    ) -> str:
        """
        Save output audio to file.

        Args:
            audio: Audio waveform to save.
            output_path: Path to save the audio file.
            sample_rate: Sample rate of the audio. Uses output_sample_rate if None.

        Returns:
            The output path.
        """
        sr = sample_rate or self.output_sample_rate

        try:
            import soundfile as sf
            sf.write(output_path, audio, sr)
        except ImportError:
            # Fallback: save as raw WAV using wave module
            import wave
            import struct

            # Normalize to int16
            audio_int16 = (audio * 32767).astype(np.int16)

            with wave.open(output_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())

        logger.info("Saved output audio to %s", output_path)
        return output_path

    def __repr__(self) -> str:
        status = "loaded" if self._model_loaded else "not loaded"
        return (
            f"RVCInference(model={self._model_path}, status={status}, "
            f"device={self.device}, sr={self.sample_rate}->{self.output_sample_rate})"
        )
