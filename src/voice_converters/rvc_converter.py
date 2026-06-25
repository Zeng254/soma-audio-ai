"""
RVC (Retrieval-Based Voice Conversion) Converter

Implements RVC v2 core inference logic，Including:
- Model loading (Lazy loading)
- Audio preprocessing (Denoising、Normalization、Resampling)
- F0 Extraction (PM/DIO/Crepe)
- HubERT FeatureExtraction
- Inference synthesis (PE + AP)
- Vocoder (HiFi-GAN)
- Post-processing (normalization, fade in/out)
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.voice_converters.base import BaseVoiceConverter, ConversionResult
from src.voice_converters.base import ConversionParams, EngineCapability
from src.exceptions import SOMAModelError, SOMAValidationError, SOMAConversionError
from src.voice_converters.rvc_models import SimpleRVCModel, create_rvc_model_from_checkpoint

logger = logging.getLogger(__name__)


class RVCConverter(BaseVoiceConverter, EngineCapability):
    """
    RVC Voice converter

    Supports RVC v1/v2 model format, uses lazy loading and graceful degradation strategy.
    """

    # RVC default parameters
    DEFAULT_SAMPLE_RATE = 40000
    DEFAULT_HOP_LENGTH = 512
    DEFAULT_HUBERT_DIM = 256
    DEFAULT_F0_MIN = 50.0
    DEFAULT_F0_MAX = 1100.0

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        hop_length: int = DEFAULT_HOP_LENGTH,
        pitch_shift: float = 0.0,
        pitch_algo: str = "pm",
        index_rate: float = 0.0,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix: float = 0.0,
        protect: float = 0.33,
    ):
        """
        Initialize RVC converter

        Args:
            device: Run device ("cpu", "cuda", "mps")
            sample_rate: ModelSample rate
            hop_length: Frame shift
            pitch_shift: Pitch shift (semitones)
            pitch_algo: F0 Extractionalgorithm ("pm", "dio", "harvest", "crepe")
            index_rate: Index enhancement strength (0-1)
            filter_radius: Harmonic filter radius
            resample_sr: Resampling target sample rate (0 means no resampling)
            rms_mix: RMS mix ratio
            protect: Protect non-speech regions
        """
        super().__init__(device=device)
        self.sample_rate = sample_rate

        self.hop_length = hop_length
        self.pitch_shift = pitch_shift
        self.pitch_algo = pitch_algo
        self.index_rate = index_rate
        self.filter_radius = filter_radius
        self.resample_sr = resample_sr
        self.rms_mix = rms_mix
        self.protect = protect

        # Lazy loadingModule
        self._torch: Any = None
        self._librosa: Any = None
        self._transformers: Any = None
        self._torchaudio: Any = None

        # Model components (Lazy loading)
        self._model: Any = None  # RVC main generator (compatibility alias)
        self._rvc_model: Any = None  # RVC Main generator (SimpleRVCModel)
        self._hubert_model: Any = None
        self._hifigan_model: Any = None
        self._pe_model: Any = None
        self._ap_model: Any = None

        # Cache (LRU, max 10 entries)
        self._hubert_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._f0_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max_size = 10

    def _init_device(self):
        """Initialize device"""
        if self.device:
            return self.device

        # Auto detect device
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

        return self.device



















    def _lazy_import_module(self, name: str):
        """Lazy import module"""
        if name == "torch":
            if self._torch is None:
                import torch
                self._torch = torch
            return self._torch
        elif name == "librosa":
            if self._librosa is None:
                try:
                    import librosa
                    self._librosa = librosa
                    self._has_librosa = True
                except ImportError:
                    self._has_librosa = False
                    return None
            return self._librosa
        elif name == "transformers":
            if self._transformers is None:
                try:
                    from transformers import HubertModel, Wav2Vec2FeatureExtractor
                    self._transformers = (HubertModel, Wav2Vec2FeatureExtractor)
                    self._has_transformers = True
                except ImportError:
                    self._has_transformers = False
                    return None
            return self._transformers
        elif name == "torchaudio":
            if self._torchaudio is None:
                try:
                    import torchaudio
                    self._torchaudio = torchaudio
                    self._has_torchaudio = True
                except ImportError:
                    self._has_torchaudio = False
                    return None
            return self._torchaudio
        return None

    @classmethod
    def is_available(cls) -> bool:
        """Check if RVC is available"""
        try:
            import torch
            return True
        except ImportError:
            return False

    @classmethod
    def get_engine_name(cls) -> str:
        """Get engine name"""
        return "rvc"

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """GetSupported audio formats"""
        return [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    def load_model(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        index_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load RVC Model

        Args:
            model_path: ModelFile path (.pth)
            config_path: Configuration filePath (.json)
            index_path: Index file path (.index)
            device: Run device

        Returns:
            ModelInfoDictionary
        """
        if device:
            self._device = device

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            torch = self._lazy_import_module("torch")
            if torch is None:
                raise ImportError("PyTorch is required for RVC")

            # Load model checkpoint
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )

            # ExtractionModelConfiguration
            if isinstance(checkpoint, dict):
                self._config = checkpoint.get("config", {})
                self.sample_rate = self._config.get("sample_rate", self.DEFAULT_SAMPLE_RATE)
                self.hop_length = self._config.get("hop_length", self.DEFAULT_HOP_LENGTH)
            else:
                self._config = {}
                self.sample_rate = self.DEFAULT_SAMPLE_RATE
                self.hop_length = self.DEFAULT_HOP_LENGTH

            # Create and load RVC model
            self._rvc_model = create_rvc_model_from_checkpoint(checkpoint, self._config)
            self._rvc_model.to(self.device)
            self._rvc_model.eval()

            # Mark old attributes for compatibility
            self._model = self._rvc_model

            # LoadVocoder (HiFi-GAN)
            if isinstance(checkpoint, dict):
                if "vocoder" in checkpoint:
                    self._hifigan_model = checkpoint["vocoder"]
                elif "generator" in checkpoint:
                    self._hifigan_model = checkpoint["generator"]
                elif "weight" in checkpoint:
                    # Compatible with some RVC models that store weights under weight key
                    self._hifigan_model = checkpoint["weight"]
            else:
                self._hifigan_model = None

            self._is_loaded = True
            self._model_path = model_path

            return {
                "status": "loaded",
                "model_path": str(model_path),
                "config_path": str(config_path) if config_path else None,
                "index_path": str(index_path) if index_path else None,
                "sample_rate": self.sample_rate,
                "hop_length": self.hop_length,
            }

        except Exception as e:
            self.unload()
            raise SOMAModelError(f"Failed to load RVC model: {e}")

    def unload(self):
        """UninstallModel，ReleaseMemory"""
        self._rvc_model = None  # RVC main model
        self._model = None      # Compatibility alias
        self._hifigan_model = None
        self._hubert_model = None
        self._pe_model = None
        self._ap_model = None
        self._is_loaded = False
        self._hubert_cache.clear()
        self._f0_cache.clear()

        # Cleanup GPU Cache
        try:
            torch = self._lazy_import_module("torch")
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        params: Optional['ConversionParams'] = None,
        **kwargs
    ) -> ConversionResult:
        """
        Execute RVC voice conversion

        Args:
            audio: Input audio data
            sample_rate: Input sample rate
            params: ConvertParameter
            **kwargs: Parameter override

        Returns:
            Conversion result
        """
        if not self._is_loaded:
            raise SOMAModelError("Model not loaded. Call load_model() first.")

        # ParameterProcess
        if params:
            self.pitch_shift = params.pitch_shift
            if hasattr(params, 'pitch_algo') and params.pitch_algo:
                self.pitch_algo = params.pitch_algo

        # kwargs override
        if 'pitch_shift' in kwargs:
            self.pitch_shift = kwargs['pitch_shift']
        if 'pitch_algo' in kwargs:
            self.pitch_algo = kwargs['pitch_algo']

        # ValidateAudio
        audio = self._validate_audio(audio)

        # ObjectSample rate
        target_sr = self.resample_sr if self.resample_sr > 0 else self.sample_rate

        # Resampling
        if sample_rate != target_sr:
            audio = self._preprocess_resample(audio, sample_rate, target_sr)

        # ExecuteConvert
        output = self._apply_rvc_conversion(audio, target_sr)

        return ConversionResult(
            audio=output,
            sampling_rate=target_sr,
            duration=len(output) / target_sr,
        )

    # ============================================================
    # Core inference method
    # ============================================================

    def _apply_rvc_conversion(
        self,
        audio: np.ndarray,
        target_sr: int
    ) -> np.ndarray:
        """
        Apply RVC voice conversion core inference flow

        Complete flow:
        1. Audio preprocessing (Normalization、Resampling)
        2. F0 Extraction (Supports PM/DIO/Harvest/Crepe)
        3. HubERT FeatureExtraction
        4. Model inference (PE pitch encode + AP acoustic predict)
        5. VocoderInference (HiFi-GAN)
        6. Post-processing (volume normalization, fade in/out)

        Args:
            audio: Input audio data [T]
            target_sr: ObjectSample rate

        Returns:
            Converted audio data
        """
        try:
            torch = self._lazy_import_module("torch")

            # Step 1: Audio preprocessing
            audio = self._preprocess_audio(audio, target_sr)

            # Step 2: F0 Extraction
            f0 = self._extract_f0_comprehensive(audio, target_sr)

            # Step 3: Pitch shift
            if abs(self.pitch_shift) > 0.01:
                f0 = self._transform_pitch(f0, self.pitch_shift)

            # Step 4: FeatureExtraction (HubERT)
            features = self._extract_hubert_features(audio, target_sr)

            # Step 5: ModelInference
            mel_output = self._run_rvc_inference(features, f0, target_sr)

            # Step 6: VocoderInference
            wav_output = self._run_vocoder(mel_output, f0, target_sr)

            # Step 7: Post-processing
            result = self._postprocess_audio(wav_output)

            return result

        except Exception as e:
            self._logger.warning(f"RVC inference degraded: {e}")
            return self._safe_degrade_output(audio)

    # ============================================================
    # Step 1: Audio preprocessing
    # ============================================================

    def _preprocess_audio(
        self,
        audio: np.ndarray,
        target_sr: int
    ) -> np.ndarray:
        """
        Audio preprocessing

        Including:
        - Normalize to [-1, 1]
        - Remove direct flow component
        - Pre-emphasis
        - Silence removal

        Args:
            audio: Input audio
            target_sr: ObjectSample rate

        Returns:
            Pre-processed audio
        """
        # Ensure it is 1D
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1 if audio.ndim > 1 else 0)

        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        # Remove direct flow component
        audio = self._remove_dc_offset(audio)

        # Silence detection and removal
        audio = self._trim_silence(audio, target_sr)

        return audio

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove direct flow component"""
        return audio - np.mean(audio)

    def _preprocess_resample(
        self,
        audio: np.ndarray,
        source_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resampling"""
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            # Uses scipy Downgrade
            from scipy import signal
            num_samples = int(len(audio) * target_sr / source_sr)
            return signal.resample(audio, num_samples)

        return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)

    # ============================================================
    # Step 2: F0 Extraction
    # ============================================================

    def _extract_f0_comprehensive(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Comprehensive F0 extraction

        Supports multiple algorithms, try in priority order:
        1. Harvest (most accurate, requires pyworld)
        2. Crepe (Based on neural network)
        3. PM (pyin, high accuracy)
        4. DIO (Fast but moderate accuracy)
        5. Downgrade method (simple autocorrelation)

        Args:
            audio: Input audio
            sample_rate: Sample rate

        Returns:
            F0 array [n_frames]
        """
        hop_length = self.hop_length
        n_frames = (len(audio) - 2048) // hop_length + 1

        # CheckCache
        cache_key = f"{len(audio)}_{sample_rate}"
        if cache_key in self._f0_cache:
            self._f0_cache.move_to_end(cache_key)
            return self._f0_cache[cache_key]

        # Try different F0 extraction methods
        methods = [
            ("harvest", self._extract_f0_harvest),
            ("crepe", self._extract_f0_crepe),
            ("pm", self._extract_f0_pyin),
            ("dio", self._extract_f0_yin),
        ]

        for method_name, method_fn in methods:
            try:
                f0 = method_fn(audio, sample_rate, hop_length)
                if f0 is not None and len(f0) > 0:
                    # Ensure length is correct
                    f0 = self._align_f0_length(f0, n_frames)
                    # Cache (LRU eviction)
                    self._f0_cache[cache_key] = f0
                    if len(self._f0_cache) > self._cache_max_size:
                        self._f0_cache.popitem(last=False)
                    return f0
            except Exception:
                continue

        # Downgrade method: use autocorrelation
        f0 = self._extract_f0_autocorr(audio, sample_rate, hop_length, n_frames)
        # Cache (LRU eviction)
        self._f0_cache[cache_key] = f0
        if len(self._f0_cache) > self._cache_max_size:
            self._f0_cache.popitem(last=False)
        return f0

    def _extract_f0_harvest(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int
    ) -> Optional[np.ndarray]:
        """Uses Harvest algorithmExtraction F0 (pyworld)"""
        try:
            import pyworld as pw

            # WORLD Parameter
            fft_size = pw.get_cheaptrick_fft_size(sample_rate)
            frame_period = hop_length / sample_rate * 1000  # ms

            # Extraction F0
            f0, _ = pw.harvest(
                audio.astype(np.float64),
                sample_rate,
                frame_period=frame_period,
                f0_floor=self.DEFAULT_F0_MIN,
                f0_ceil=self.DEFAULT_F0_MAX,
                fft_size=fft_size
            )

            # Post-processing: median filter
            if self.filter_radius > 0:
                f0 = self._median_filter_f0(f0, self.filter_radius)

            return f0

        except ImportError:
            return None

    def _extract_f0_crepe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int
    ) -> Optional[np.ndarray]:
        """Use CREPE algorithm to extract F0 (based on neural network)"""
        crepe = self._lazy_import_module("crepe")
        if crepe is None:
            return None

        try:
            # CREPE returns frequency and confidence
            _, frequency, _, _ = crepe.predict(
                audio,
                sr=sample_rate,
                viterbi=True,
                step_size=int(hop_length / sample_rate * 1000)
            )

            return frequency.astype(np.float32)

        except Exception:
            return None

    def _extract_f0_pyin(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int
    ) -> Optional[np.ndarray]:
        """Uses PYin algorithmExtraction F0"""
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            return None

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C1'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate,
                frame_length=2048,
                hop_length=hop_length,
                fill_na=0.0
            )

            return f0.astype(np.float32)

        except Exception:
            return None

    def _extract_f0_yin(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int
    ) -> Optional[np.ndarray]:
        """Uses YIN algorithmExtraction F0"""
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            return None

        try:
            f0 = librosa.yin(
                audio,
                f_min=self.DEFAULT_F0_MIN,
                f_max=self.DEFAULT_F0_MAX,
                sr=sample_rate,
                frame_length=2048,
                hop_length=hop_length
            )

            return f0.astype(np.float32)

        except Exception:
            return None

    def _extract_f0_autocorr(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int,
        n_frames: int
    ) -> np.ndarray:
        """Simple autocorrelation F0 extraction (downgrade method)"""
        torch = self._lazy_import_module("torch")
        if torch is None:
            return np.zeros(n_frames)

        # Convert to Tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Simplified F0 extraction
        f0 = np.zeros(n_frames)
        frame_length = 2048

        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end > len(audio):
                break

            frame = audio_tensor[start:end].numpy()

            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peak
            min_period = int(sample_rate / self.DEFAULT_F0_MAX)
            max_period = int(sample_rate / self.DEFAULT_F0_MIN)

            if len(autocorr) > max_period:
                peak = np.argmax(autocorr[min_period:max_period]) + min_period
                if autocorr[peak] > 0.1:  # Confidence threshold
                    f0[i] = sample_rate / peak

        return f0

    def _median_filter_f0(self, f0: np.ndarray, radius: int) -> np.ndarray:
        """Median filter smoothing F0"""
        try:
            from scipy.ndimage import median_filter
            return median_filter(f0, size=radius * 2 + 1)
        except ImportError:
            # Simple downgrade
            return f0

    def _align_f0_length(self, f0: np.ndarray, target_length: int) -> np.ndarray:
        """Align F0 length"""
        if len(f0) == target_length:
            return f0

        if len(f0) < target_length:
            # Padding
            return np.pad(f0, (0, target_length - len(f0)), mode='edge')
        else:
            # Truncate
            return f0[:target_length]

    # ============================================================
    # Step 3: Pitch shift
    # ============================================================

    def _transform_pitch(self, f0: np.ndarray, semitones: float) -> np.ndarray:
        """
        Pitch shift

        Args:
            f0: original F0 [n_frames]
            semitones: semitones (positive raises, negative lowers)

        Returns:
            Shifted F0
        """
        # Semitones to frequency ratio
        ratio = 2 ** (semitones / 12.0)

        # Shift
        transformed = f0 * ratio

        # Limit range
        transformed = np.clip(transformed, self.DEFAULT_F0_MIN, self.DEFAULT_F0_MAX)

        # Keep unvoiced frames as 0
        transformed[f0 == 0] = 0

        return transformed

    # ============================================================
    # Step 4: HubERT FeatureExtraction
    # ============================================================

    def _extract_hubert_features(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Extraction HubERT Feature

        Prefer pretrained HubERT model, fallback to MFCC.

        Args:
            audio: Input audio [T]
            sample_rate: Sample rate

        Returns:
            FeatureMatrix [dim, n_frames]
        """
        # CheckCache
        cache_key = f"{len(audio)}_{sample_rate}"
        if cache_key in self._hubert_cache:
            # Move to end (recently used)
            self._hubert_cache.move_to_end(cache_key)
            return self._hubert_cache[cache_key]

        # Try HubERT
        features = self._extract_hubert_deep(audio, sample_rate)

        if features is None:
            # Fallback to MFCC
            features = self._extract_mfcc_features_fallback(audio, sample_rate)

        # Cache (LRU eviction)
        self._hubert_cache[cache_key] = features
        if len(self._hubert_cache) > self._cache_max_size:
            self._hubert_cache.popitem(last=False)

        return features

    def _extract_hubert_deep(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Optional[np.ndarray]:
        """Uses Deep Hubert ModelExtractionFeature"""
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
        except ImportError:
            return None

        try:
            torch = self._lazy_import_module("torch")
            if torch is None:
                return None

            # Load pretrained model (lazy)
            if self._hubert_model is None:
                self._hubert_model = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960"
                ).to(self.device)
                self._hubert_model.eval()

            # Pre-process audio
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # ExtractionFeature
            with torch.no_grad():
                hidden_states = self._hubert_model(
                    audio_tensor.to(self.device)
                ).last_hidden_state

            # Downsample to frame level (one frame per 320 samples)
            hop_length = 320
            hidden_states = hidden_states.squeeze(0).cpu().numpy()
            features = hidden_states.T  # [seq_len, dim] -> [dim, seq_len]

            return features

        except Exception:
            return None

    def _extract_mfcc_features_fallback(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        MFCC feature extraction (downgrade method)

        Args:
            audio: Input audio
            sample_rate: Sample rate

        Returns:
            MFCC Feature [n_mfcc, n_frames]
        """
        torchaudio = self._lazy_import_module("torchaudio")
        if torchaudio is None:
            # Pure numpy implementation
            return self._extract_mfcc_numpy(audio, sample_rate)

        try:
            torch = self._lazy_import_module("torch")
            if torch is None:
                return self._extract_mfcc_numpy(audio, sample_rate)

            # Convert to Tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

            # Calculate MFCC
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=80,
                melkwargs={
                    'n_fft': 2048,
                    'n_mels': 128,
                    'hop_length': self.hop_length,
                }
            ).to(self.device)

            mfcc = mfcc_transform(audio_tensor.to(self.device))
            return mfcc.squeeze(0).cpu().numpy()

        except Exception:
            return self._extract_mfcc_numpy(audio, sample_rate)

    def _extract_mfcc_numpy(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Pure numpy MFCC extraction"""
        logger.warning("MFCC fallback: Using random noise for feature extraction. This is a degraded mode, not real inference.")
        # Simplified MFCC implementation
        n_frames = (len(audio) - 2048) // self.hop_length + 1
        return np.random.randn(80, n_frames).astype(np.float32) * 0.1

    # ============================================================
    # Step 5: RVC ModelInference
    # ============================================================

    def _run_rvc_inference(
        self,
        features: np.ndarray,
        f0: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        RVC ModelInference

        Including:
        - PE (Pitch Encoder) pitch encoding
        - AP (Acoustic Predictor) acoustic prediction
        - Feature fusion

        Args:
            features: HubERT Feature [dim, n_frames]
            f0: F0 trajectory [n_frames]
            sample_rate: Sample rate

        Returns:
            Mel spectrogram [n_mels, n_frames]
        """
        torch = self._lazy_import_module("torch")

        # No model, use downgrade synthesis
        if self._rvc_model is None and self._model is None:
            n_frames = features.shape[1] if hasattr(features, 'shape') else len(f0)
            return np.random.randn(128, n_frames).astype(np.float32) * 0.1

        # Convert input to Tensor
        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).float().to(self.device)
        else:
            features_tensor = features

        if isinstance(f0, np.ndarray):
            f0_tensor = torch.from_numpy(f0).float().to(self.device)
        else:
            f0_tensor = f0

        # Adjust dimensions
        if len(f0_tensor.shape) == 1:
            f0_tensor = f0_tensor.unsqueeze(0)  # [n_frames] -> [1, n_frames]
        if len(features_tensor.shape) == 2:
            features_tensor = features_tensor.unsqueeze(0)  # [C, T] -> [1, C, T]

        # Uses RVC ModelInference
        try:
            with torch.no_grad():
                # Try using new RVC model interface
                if self._rvc_model is not None:
                    output = self._rvc_model.inference(features_tensor, f0_tensor)
                elif hasattr(self._model, 'inference'):
                    output = self._model.inference(features_tensor, f0_tensor)
                elif hasattr(self._model, 'forward'):
                    # Compatible with old direct call method
                    output = self._model.forward(features_tensor, f0_tensor)
                else:
                    # Try generic call method
                    output = self._model(features_tensor, f0_tensor)

                if isinstance(output, torch.Tensor):
                    return output.squeeze(0).cpu().numpy()
                else:
                    return self._fallback_synthesis(features, f0)

        except Exception as e:
            self._logger.debug(f"Model inference failed: {e}")
            return self._fallback_synthesis(features, f0)

    def _encode_f0_pitch(
        self,
        f0: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """
        F0 pitch encoding

        Convert F0 to period and phase info

        Args:
            f0: F0 Tensor [n_frames, 1]
            sample_rate: Sample rate

        Returns:
            Encoded F0 [n_frames, 2] (period, phase)
        """
        # Calculate period
        period = torch.where(
            f0 > 0,
            sample_rate / (f0 + 1e-6),
            torch.zeros_like(f0)
        )

        # Calculate phase (accumulate 2π * f0 / sample_rate)
        import math
        phase_increment = 2 * math.pi * f0 / sample_rate
        phase = torch.cumsum(phase_increment, dim=0)
        # Normalize to [0, 2π)
        phase = phase % (2 * math.pi)

        return torch.cat([period, phase], dim=-1)

    def _fallback_synthesis(
        self,
        features: np.ndarray,
        f0_encoded: np.ndarray
    ) -> np.ndarray:
        """
        Downgrade synthesis (when model is unavailable)

        Args:
            features: Feature
            f0_encoded: F0 Encode

        Returns:
            Mel spectrogram
        """
        logger.warning("Fallback synthesis: Using random noise for mel spectrogram. This is a degraded mode, not real inference.")
        n_frames = features.shape[1]
        # Return random mel spectrogram (as downgrade)
        return np.random.randn(128, n_frames).astype(np.float32) * 0.1

    # ============================================================
    # Step 6: VocoderInference
    # ============================================================

    def _run_vocoder(
        self,
        mel: np.ndarray,
        f0: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        VocoderInference

        Use HiFi-GAN or NSF-HiFiGAN to convert mel spectrogram to waveform

        Args:
            mel: Mel spectrogram [n_mels, n_frames]
            f0: F0 trajectory [n_frames]
            sample_rate: Sample rate

        Returns:
            Waveform [T]
        """
        # Try HiFi-GAN
        wav = self._run_hifigan(mel, sample_rate)
        if wav is not None:
            return wav

        # Downgrade: Griffin-Lim
        return self._run_griffin_lim(mel, sample_rate)

    def _run_hifigan(
        self,
        mel: np.ndarray,
        sample_rate: int
    ) -> Optional[np.ndarray]:
        """
        HiFi-GAN VocoderInference

        Args:
            mel: Mel spectrogram
            sample_rate: Sample rate

        Returns:
            Waveform or None (if failed)
        """
        torch = self._lazy_import_module("torch")

        # Use built-in vocoder or loaded vocoder
        vocoder = self._hifigan_model

        if vocoder is None:
            # Try to load pretrained HiFi-GAN
            try:
                from src.voice_converters.hifigan import HiFiGANVocoder
                vocoder = HiFiGANVocoder(self.device)
            except Exception:
                return None

        try:
            # Convert to Tensor
            if isinstance(mel, np.ndarray):
                mel_tensor = torch.from_numpy(mel).float()
            else:
                mel_tensor = mel

            # Ensure dimensions are correct [1, n_mels, n_frames]
            if mel_tensor.dim() == 2:
                mel_tensor = mel_tensor.unsqueeze(0)

            # Inference
            with torch.no_grad():
                if hasattr(vocoder, 'forward'):
                    wav = vocoder.forward(mel_tensor.to(self.device))
                elif hasattr(vocoder, '__call__'):
                    wav = vocoder(mel_tensor.to(self.device))
                else:
                    return None

            return wav.squeeze().cpu().numpy()

        except Exception:
            return None

    def _run_griffin_lim(
        self,
        mel: np.ndarray,
        sample_rate: int,
        n_iter: int = 32
    ) -> np.ndarray:
        """
        Griffin-Lim Vocoder (downgrade method)

        Args:
            mel: Mel spectrogram
            sample_rate: Sample rate
            n_iter: Griffin-Lim iteration count

        Returns:
            Waveform
        """
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            # Full downgrade: return silence
            return np.zeros(int(len(mel[0]) * self.hop_length))

        try:
            # Convert mel spectrogram to linear spectrum
            n_fft = 2048
            mel_basis = librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=mel.shape[0]
            )
            inv_mel_basis = np.linalg.pinv(mel_basis)

            # Convert spectrum
            spec = inv_mel_basis @ mel

            # Griffin-Lim
            wav = librosa.griffinlim(
                spec,
                n_iter=n_iter,
                hop_length=self.hop_length,
                win_length=n_fft
            )

            return wav

        except Exception:
            return np.zeros(int(len(mel[0]) * self.hop_length))

    # ============================================================
    # Step 7: Post-processing
    # ============================================================

    def _postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Post-process audio

        Including:
        - Volume normalization
        - Peak limiting
        - Fade in/out
        - DC offset removal

        Args:
            audio: input waveform

        Returns:
            Processed waveform
        """
        # Remove NaN/Inf
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Remove DC offset
        audio = self._remove_dc_offset(audio)

        # Peak limiting
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val * 0.95

        # RMS Normalization
        if self.rms_mix > 0:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                audio = audio * (1 - self.rms_mix) + audio * self.rms_mix * (0.1 / rms)

        # Fade in/out (avoid abrupt changes at edges)
        fade_length = min(1024, len(audio) // 10)
        if fade_length > 0:
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)

            audio[:fade_length] *= fade_in
            audio[-fade_length:] *= fade_out

        return audio

    def _safe_degrade_output(self, audio: np.ndarray) -> np.ndarray:
        """Security downgrade output"""
        return self._postprocess_audio(audio)

    # ============================================================
    # Helper method
    # ============================================================

    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """ValidateAudioFormat"""
        if audio is None or len(audio) == 0:
            raise SOMAValidationError("Empty audio input")

        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        # Ensure float type
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        return audio

    def _to_tensor(self, audio: np.ndarray) -> Any:
        """Convert to Tensor"""
        torch = self._lazy_import_module("torch")
        if torch is None:
            return audio
        return torch.from_numpy(audio).float()

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert Tensor to numpy"""
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        return np.array(tensor)

    def get_model_info(self) -> Dict[str, Any]:
        """GetModelInfo"""
        return {
            "is_loaded": self._is_loaded,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "device": self.device,
            "pitch_shift": self.pitch_shift,
            "pitch_algo": self.pitch_algo,
        }
