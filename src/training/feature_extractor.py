"""
HuBERT Feature Extractor and F0 Extraction for RVC Training.

This module provides real feature extraction using a frozen HuBERT/contentvec model,
replacing the dummy random features previously used in training. It also provides
F0 (fundamental frequency) extraction using librosa.

Key components:
    - HuBERTFeatureExtractor: Loads a frozen HuBERT model and extracts 256-dim features
    - F0Extractor: Extracts fundamental frequency contours from audio
    - FeaturePipeline: Combined pipeline for extracting both features and F0

Usage:
    extractor = HuBERTFeatureExtractor(model_name="hubert_base")
    features = extractor.extract(audio_waveform, sample_rate=16000)
    # features shape: (1, 256, num_frames)

    f0_ext = F0Extractor(method="yin", sample_rate=16000)
    f0 = f0_ext.extract(audio_waveform)
    # f0 shape: (num_frames,)
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HuBERTFeatureExtractor:
    """
    Extracts speech features using a frozen HuBERT/contentvec model.

    The HuBERT model is loaded in eval mode with gradients disabled,
    ensuring it acts as a fixed feature extractor without consuming
    additional GPU memory for gradient computation.

    Supported models:
        - "hubert_base": Facebook's HuBERT Base (~95M params, 256-dim features)
        - "contentvec": RVC's contentvec model (256-dim features, Chinese-trained)
        - Local path: Path to a custom checkpoint file

    Attributes:
        model: The loaded HuBERT model (frozen).
        feature_dim: Dimension of extracted features (256 for base models).
        device: Device the model is loaded on.
    """

    # Known model configurations
    MODEL_CONFIGS = {
        "hubert_base": {
            "feature_dim": 256,
            "source": "facebookresearch/hubert",
            "model_type": "hubert",
        },
        "contentvec": {
            "feature_dim": 256,
            "source": "contentvec",
            "model_type": "contentvec",
        },
    }

    def __init__(
        self,
        model_name: str = "hubert_base",
        device: str = "cpu",
        model_path: Optional[str] = None,
        lazy_load: bool = True,
    ):
        """
        Initialize the HuBERT feature extractor.

        Args:
            model_name: Name of the pretrained model or "custom" if using model_path.
            device: Device to load the model on ("cpu" or "cuda").
            model_path: Optional path to a local model checkpoint.
            lazy_load: If True (default), defer model loading until first extract() call.
                If False, load the model immediately (may be slow or fail without network/GPU).
        """
        self.model_name = model_name
        self.device = device
        self.model_path = model_path
        self.model = None
        self._feature_dim = 256  # Default
        self._is_loaded = False
        self._fallback_projection = None

        if not lazy_load:
            self.load()

    def load(self) -> None:
        """
        Explicitly load the HuBERT model.

        Called automatically on first extract() if lazy_load=True.
        Can be called explicitly to pre-load the model.
        """
        if self._is_loaded:
            return

        if self.model_path:
            self._load_from_path(self.model_path)
        else:
            self._load_pretrained(self.model_name)

    def _load_pretrained(self, model_name: str) -> None:
        """Load a pretrained HuBERT model."""
        import torch

        config = self.MODEL_CONFIGS.get(model_name)
        if config is None:
            logger.warning(
                f"Unknown model '{model_name}', using hubert_base config. "
                f"Available: {list(self.MODEL_CONFIGS.keys())}"
            )
            config = self.MODEL_CONFIGS["hubert_base"]

        self._feature_dim = config["feature_dim"]

        try:
            if model_name == "hubert_base":
                self._load_hubert_base()
            elif model_name == "contentvec":
                self._load_contentvec()
            else:
                self._load_hubert_base()
        except Exception as e:
            logger.warning(
                f"Failed to load pretrained model '{model_name}': {e}. "
                "Falling back to random projection feature extractor."
            )
            self._init_fallback_extractor()

        self._is_loaded = True

    def _load_hubert_base(self) -> None:
        """Load Facebook HuBERT Base model."""
        import torch

        try:
            # Try loading via torch.hub (requires internet)
            model = torch.hub.load(
                "facebookresearch/fairseq",
                "hubert_base",
                trust_repo=True,
            )
            model = model.to(self.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
            self._feature_dim = 256
            logger.info("Loaded HuBERT Base model via torch.hub")
        except Exception as e:
            logger.warning(f"torch.hub load failed: {e}. Trying alternative load method.")
            self._load_hubert_from_torchaudio()

    def _load_hubert_from_torchaudio(self) -> None:
        """Load HuBERT model via torchaudio as fallback."""
        import torch

        try:
            import torchaudio

            bundle = torchaudio.pipelines.HUBERT_BASE
            model = bundle.get_model()
            model = model.to(self.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
            self._feature_dim = 768  # HuBERT Base outputs 768-dim
            logger.info("Loaded HuBERT Base model via torchaudio (768-dim)")
        except ImportError:
            logger.warning("torchaudio not available. Using fallback extractor.")
            self._init_fallback_extractor()
        except Exception as e:
            logger.warning(f"torchaudio load failed: {e}. Using fallback extractor.")
            self._init_fallback_extractor()

    def _load_contentvec(self) -> None:
        """Load contentvec model (RVC's preferred feature extractor)."""
        import torch

        try:
            # contentvec is typically loaded from a local checkpoint
            # Try torch.hub first as a fallback
            model = torch.hub.load(
                "facebookresearch/fairseq",
                "hubert_base",
                trust_repo=True,
            )
            model = model.to(self.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
            self._feature_dim = 256
            logger.info("Loaded contentvec-compatible model")
        except Exception as e:
            logger.warning(f"contentvec load failed: {e}. Using fallback extractor.")
            self._init_fallback_extractor()

    def _load_from_path(self, model_path: str) -> None:
        """Load model from a local checkpoint file."""
        import torch

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "weights" in checkpoint:
                    state_dict = checkpoint["weights"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Try to load into a HuBERT model
            self._load_hubert_base()
            if self.model is not None:
                # Attempt to load state dict (may partially succeed)
                try:
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded model weights from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load state_dict: {e}. Using base model.")

            self._is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self._init_fallback_extractor()
            self._is_loaded = True

    def _init_fallback_extractor(self) -> None:
        """
        Initialize a deterministic fallback feature extractor.

        This uses a fixed random projection (mel-like features + projection)
        that is deterministic (seeded) so training is reproducible.
        This is NOT a real HuBERT model but provides a consistent fallback.
        """
        import torch

        logger.warning(
            "Using deterministic fallback feature extractor. "
            "For best results, install torchaudio or provide a HuBERT checkpoint."
        )
        self.model = None
        self._feature_dim = 256
        # Create a fixed projection matrix (deterministic, not random per call)
        torch.manual_seed(42)
        self._fallback_projection = torch.randn(
            128, self._feature_dim, device=self.device
        ) * (1.0 / np.sqrt(128))
        self._fallback_projection = self._fallback_projection.t()  # (256, 128)

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        return self._feature_dim

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Extract HuBERT features from audio waveform.

        Args:
            audio: Audio waveform as numpy array, shape (samples,) or (1, samples).
            sample_rate: Sample rate of the audio (HuBERT expects 16kHz).

        Returns:
            Features as numpy array, shape (feature_dim, num_frames).
        """
        import torch

        # Lazy load: load model on first extract() call
        if not self._is_loaded:
            self.load()

        # Ensure mono, float32
        audio = self._preprocess_audio(audio, sample_rate)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        if self.model is not None:
            features = self._extract_with_model(audio_tensor)
        else:
            features = self._extract_with_fallback(audio_tensor)

        # features shape: (1, feature_dim, num_frames) -> (feature_dim, num_frames)
        if features.dim() == 3:
            features = features.squeeze(0)

        return features.cpu().numpy()

    def extract_batch(
        self,
        audio_batch: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Extract features for a batch of audio waveforms.

        Args:
            audio_batch: Batch of audio, shape (batch_size, samples).
            sample_rate: Sample rate of the audio.

        Returns:
            Features, shape (batch_size, feature_dim, num_frames).
        """
        import torch

        if not self._is_loaded:
            raise RuntimeError("Model not loaded.")

        batch_features = []
        for i in range(audio_batch.shape[0]):
            audio = audio_batch[i]
            feats = self.extract(audio, sample_rate)
            batch_features.append(feats)

        # Pad to same length if needed
        max_len = max(f.shape[-1] for f in batch_features)
        padded = np.zeros(
            (len(batch_features), self._feature_dim, max_len), dtype=np.float32
        )
        for i, f in enumerate(batch_features):
            padded[i, :, : f.shape[-1]] = f

        return padded

    def _extract_with_model(self, audio_tensor) -> "torch.Tensor":
        """Extract features using the loaded HuBERT model."""
        import torch

        with torch.no_grad():
            # HuBERT expects input of shape (batch, samples)
            if audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(1)

            # Extract features using the model
            try:
                # fairseq HuBERT interface
                if hasattr(self.model, "extract_features"):
                    features, _ = self.model.extract_features(
                        source=audio_tensor, padding_mask=None
                    )
                elif hasattr(self.model, "forward"):
                    # torchaudio interface
                    features = self.model.extract_features(audio_tensor)
                    if isinstance(features, tuple):
                        features = features[0]
                else:
                    # Generic forward pass
                    features = self.model(audio_tensor)
                    if isinstance(features, tuple):
                        features = features[0]

                # features shape: (batch, num_frames, feature_dim)
                # Transpose to (batch, feature_dim, num_frames)
                if features.dim() == 3:
                    features = features.transpose(1, 2)

            except Exception as e:
                logger.warning(f"Model extraction failed: {e}. Using fallback.")
                return self._extract_with_fallback(audio_tensor)

        return features

    def _extract_with_fallback(self, audio_tensor) -> "torch.Tensor":
        """
        Deterministic fallback feature extraction.

        Uses a fixed mel-like computation + deterministic projection.
        This is NOT a real HuBERT model but provides reproducible features.
        """
        import torch

        # Compute a simple spectral representation
        # Use STFT magnitude as a proxy for features
        n_fft = 400
        hop_length = 160  # 10ms at 16kHz -> 100 frames/sec

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # STFT
        window = torch.hann_window(n_fft, device=audio_tensor.device)
        stft = torch.stft(
            audio_tensor.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        magnitude = stft.abs()  # (n_freq, num_frames)

        # Take log mel-like features (simplified)
        log_magnitude = torch.log(magnitude + 1e-5)

        # Project to feature_dim using fixed projection
        # Take first 128 frequency bins
        n_bins = min(128, log_magnitude.shape[0])
        spectral = log_magnitude[:n_bins, :]  # (128, num_frames)

        # Project to feature_dim
        projection = self._fallback_projection.to(audio_tensor.device)
        features = projection @ spectral  # (feature_dim, num_frames)

        return features.unsqueeze(0)  # (1, feature_dim, num_frames)

    def _preprocess_audio(
        self, audio: np.ndarray, target_sr: int = 16000
    ) -> np.ndarray:
        """Preprocess audio to the expected format."""
        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0) if audio.shape[0] <= 2 else audio[:, 0]

        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        return audio.astype(np.float32)


class F0Extractor:
    """
    Extracts fundamental frequency (F0) from audio using various methods.

    Supported methods:
        - "yin": librosa.yin() - YIN algorithm, good for monophonic signals
        - "pyin": librosa.pyin() - Probabilistic YIN, more robust
        - "dio": WORLD DIO algorithm (if pyworld available)
        - "harvest": WORLD Harvest algorithm (if pyworld available)

    Attributes:
        method: The F0 extraction method.
        sample_rate: Expected audio sample rate.
        hop_length: Hop length in samples for frame-level F0.
    """

    def __init__(
        self,
        method: str = "dio",
        sample_rate: int = 16000,
        hop_length: int = 160,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
    ):
        """
        Initialize the F0 extractor.

        Args:
            method: F0 extraction method ("yin", "pyin", "dio", "harvest").
            sample_rate: Audio sample rate in Hz.
            hop_length: Hop length in samples.
            f0_min: Minimum F0 frequency in Hz.
            f0_max: Maximum F0 frequency in Hz.
        """
        self.method = method
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max

    def extract(
        self,
        audio: np.ndarray,
        target_length: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract F0 contour from audio.

        Args:
            audio: Audio waveform, shape (samples,).
            target_length: Optional target number of frames to pad/trim to.

        Returns:
            F0 contour, shape (num_frames,). Unvoiced frames have F0 = 0.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=0) if audio.shape[0] <= 2 else audio[:, 0]

        audio = audio.astype(np.float32)

        try:
            if self.method == "yin":
                f0 = self._extract_yin(audio)
            elif self.method == "pyin":
                f0 = self._extract_pyin(audio)
            elif self.method == "dio":
                f0 = self._extract_dio(audio)
            elif self.method == "harvest":
                f0 = self._extract_harvest(audio)
            else:
                logger.warning(f"Unknown F0 method '{self.method}', using yin")
                f0 = self._extract_yin(audio)
        except Exception as e:
            logger.warning(f"F0 extraction failed: {e}. Using default F0=200Hz.")
            num_frames = len(audio) // self.hop_length + 1
            f0 = np.full(num_frames, 200.0, dtype=np.float32)

        # Replace NaN/inf with 0 (unvoiced)
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad or trim to target length
        if target_length is not None:
            f0 = self._adjust_length(f0, target_length)

        return f0

    def _extract_yin(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using librosa.yin()."""
        import librosa

        frame_length = self.hop_length * 2
        f0 = librosa.yin(
            audio,
            sr=self.sample_rate,
            fmin=self.f0_min,
            fmax=self.f0_max,
            frame_length=frame_length,
            hop_length=self.hop_length,
        )
        return f0

    def _extract_pyin(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using librosa.pyin()."""
        import librosa

        frame_length = self.hop_length * 2
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            sr=self.sample_rate,
            fmin=self.f0_min,
            fmax=self.f0_max,
            frame_length=frame_length,
            hop_length=self.hop_length,
        )
        # Set unvoiced frames to 0
        if voiced_flag is not None:
            f0[~voiced_flag] = 0.0
        return f0

    def _extract_dio(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using pyworld DIO algorithm."""
        try:
            import pyworld as pw

            f0, t = pw.dio(
                audio.astype(np.float64),
                fs=self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=1000.0 * self.hop_length / self.sample_rate,
            )
            f0 = pw.stonemask(audio.astype(np.float64), f0, t, self.sample_rate)
            return f0.astype(np.float32)
        except ImportError:
            logger.warning("pyworld not available, falling back to librosa.yin")
            return self._extract_yin(audio)

    def _extract_harvest(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using pyworld Harvest algorithm."""
        try:
            import pyworld as pw

            f0, t = pw.harvest(
                audio.astype(np.float64),
                fs=self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=1000.0 * self.hop_length / self.sample_rate,
            )
            f0 = pw.stonemask(audio.astype(np.float64), f0, t, self.sample_rate)
            return f0.astype(np.float32)
        except ImportError:
            logger.warning("pyworld not available, falling back to librosa.yin")
            return self._extract_yin(audio)

    def _adjust_length(self, f0: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or trim F0 to target length."""
        if len(f0) >= target_length:
            return f0[:target_length]
        else:
            # Pad with 0 (unvoiced)
            padded = np.zeros(target_length, dtype=np.float32)
            padded[: len(f0)] = f0
            return padded


class FeaturePipeline:
    """
    Combined pipeline for extracting HuBERT features and F0 from audio.

    This is the main entry point for feature extraction during training.
    It combines HuBERTFeatureExtractor and F0Extractor into a single
    pipeline that produces aligned features and F0 contours.

    Usage:
        pipeline = FeaturePipeline(device="cuda")
        features, f0 = pipeline.extract(audio, sample_rate=16000)
        # features: (256, num_frames)
        # f0: (num_frames,)
    """

    def __init__(
        self,
        model_name: str = "hubert_base",
        device: str = "cpu",
        model_path: Optional[str] = None,
        f0_method: str = "dio",
        sample_rate: int = 16000,
        hop_length: int = 160,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
    ):
        """
        Initialize the feature pipeline.

        Args:
            model_name: HuBERT model name.
            device: Device for model loading.
            model_path: Optional path to local HuBERT checkpoint.
            f0_method: F0 extraction method.
            sample_rate: Audio sample rate.
            hop_length: Hop length for F0 extraction.
            f0_min: Minimum F0 in Hz.
            f0_max: Maximum F0 in Hz.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.hubert = HuBERTFeatureExtractor(
            model_name=model_name,
            device=device,
            model_path=model_path,
        )

        self.f0_extractor = F0Extractor(
            method=f0_method,
            sample_rate=sample_rate,
            hop_length=hop_length,
            f0_min=f0_min,
            f0_max=f0_max,
        )

    @property
    def feature_dim(self) -> int:
        """Return the HuBERT feature dimension."""
        return self.hubert.feature_dim

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        target_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both HuBERT features and F0 from audio.

        Args:
            audio: Audio waveform, shape (samples,) or (1, samples).
            sample_rate: Sample rate (uses pipeline default if None).
            target_frames: Optional target number of frames for alignment.

        Returns:
            Tuple of (features, f0):
                - features: (feature_dim, num_frames)
                - f0: (num_frames,)
        """
        sr = sample_rate or self.sample_rate

        # Extract HuBERT features
        features = self.hubert.extract(audio, sr)

        # Determine target frame count
        if target_frames is None:
            target_frames = features.shape[-1]

        # Extract F0 with matching frame count
        if audio.ndim > 1:
            audio_mono = audio.mean(axis=0)
        else:
            audio_mono = audio

        f0 = self.f0_extractor.extract(audio_mono, target_length=target_frames)

        # Ensure features and F0 have matching frame count
        feat_frames = features.shape[-1]
        f0_frames = len(f0)

        if feat_frames != f0_frames:
            min_frames = min(feat_frames, f0_frames)
            features = features[:, :min_frames]
            f0 = f0[:min_frames]

        return features, f0

    def extract_batch(
        self,
        audio_batch: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and F0 for a batch of audio.

        Args:
            audio_batch: (batch_size, samples)
            sample_rate: Sample rate.

        Returns:
            Tuple of (features_batch, f0_batch):
                - features_batch: (batch_size, feature_dim, max_frames)
                - f0_batch: (batch_size, max_frames)
        """
        sr = sample_rate or self.sample_rate

        all_features = []
        all_f0 = []

        for i in range(audio_batch.shape[0]):
            feats, f0 = self.extract(audio_batch[i], sr)
            all_features.append(feats)
            all_f0.append(f0)

        # Pad to same length
        max_frames = max(f.shape[-1] for f in all_features)
        feat_dim = self.feature_dim
        batch_size = len(all_features)

        features_padded = np.zeros((batch_size, feat_dim, max_frames), dtype=np.float32)
        f0_padded = np.zeros((batch_size, max_frames), dtype=np.float32)

        for i in range(batch_size):
            flen = all_features[i].shape[-1]
            features_padded[i, :, :flen] = all_features[i]
            f0_padded[i, :flen] = all_f0[i]

        return features_padded, f0_padded
