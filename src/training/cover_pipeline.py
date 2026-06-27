"""
Cover Pipeline - Complete AI cover generation pipeline

Integrates audio separation, voice conversion, and audio processing
into a complete workflow for generating AI covers.

Pipeline stages:
1. Source separation (extract vocals from original song)
2. Optional: Dereverberation (clean up extracted vocals)
3. Voice conversion (convert to target voice)
4. Optional: Mix with accompaniment

Usage:
    pipeline = CoverPipeline(model_path="path/to/model.pth")
    output = pipeline.generate_cover(
        source_audio="path/to/song.wav",
        output_path="path/to/output.wav",
    )
"""

import logging
from typing import Optional, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CoverConfig:
    """Configuration for cover generation pipeline."""
    
    # Separation settings
    separate_vocals: bool = True
    separation_mode: str = "2stems"  # "2stems" or "4stems"
    separation_backend: str = "auto"  # "auto", "demucs", "msst"
    
    # Preprocessing settings
    dereverb: bool = False
    dereverb_method: str = "spectral"  # "spectral" or "wiener"
    dereverb_reduction_db: float = 10.0
    
    # Voice conversion settings
    transpose: int = 0  # Pitch shift in semitones
    f0_method: str = "harvest"  # F0 extraction method
    f0_upsampling: int = 1
    
    # Output settings
    output_sample_rate: int = 44100
    output_format: str = "wav"  # "wav", "mp3", "flac"
    
    # Mixing settings
    mix_with_accompaniment: bool = True
    vocal_volume: float = 1.0
    accompaniment_volume: float = 0.8


@dataclass
class CoverResult:
    """Result of cover generation."""
    
    output_audio: np.ndarray
    sample_rate: int
    output_path: Optional[str] = None
    
    # Intermediate results (for debugging/analysis)
    separated_vocals: Optional[np.ndarray] = None
    separated_accompaniment: Optional[np.ndarray] = None
    dereverberated_vocals: Optional[np.ndarray] = None
    converted_vocals: Optional[np.ndarray] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CoverPipeline:
    """
    Complete AI cover generation pipeline.
    
    Integrates audio separation, voice conversion, and audio processing
    into a single workflow.
    
    Attributes:
        model_path: Path to voice conversion model.
        config: Pipeline configuration.
        separator: Audio separator instance.
        inference: Voice conversion inference instance.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[CoverConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the cover pipeline.
        
        Args:
            model_path: Path to voice conversion model checkpoint.
            config: Pipeline configuration. Uses defaults if None.
            device: Device to run on ("cpu", "cuda", "mps").
        """
        self.model_path = model_path
        self.config = config or CoverConfig()
        self.device = device or self._get_default_device()
        
        # Lazy-loaded components
        self._separator = None
        self._inference = None
        
    def _get_default_device(self) -> str:
        """Get default compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    @property
    def separator(self):
        """Lazy-load audio separator."""
        if self._separator is None:
            from src.separators.audio_separator import AudioSeparator
            self._separator = AudioSeparator(
                backend=self.config.separation_backend,
                device=self.device,
            )
        return self._separator
    
    @property
    def inference(self):
        """Lazy-load voice conversion inference."""
        if self._inference is None:
            if self.model_path is None:
                raise ValueError("model_path required for voice conversion")
            from src.training.inference import RVCInference
            self._inference = RVCInference(
                model_path=self.model_path,
                device=self.device,
            )
        return self._inference
    
    def generate_cover(
        self,
        source_audio: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        config: Optional[CoverConfig] = None,
    ) -> CoverResult:
        """
        Generate AI cover from source audio.
        
        Pipeline:
        1. Load source audio
        2. Separate vocals (if enabled)
        3. Dereverberate vocals (if enabled)
        4. Convert voice
        5. Mix with accompaniment (if enabled)
        6. Save output
        
        Args:
            source_audio: Path to source audio file or numpy array.
            output_path: Path to save output audio. If None, returns without saving.
            config: Override default configuration for this run.
            
        Returns:
            CoverResult with output audio and metadata.
        """
        cfg = config or self.config
        result = CoverResult(
            output_audio=None,
            sample_rate=cfg.output_sample_rate,
            metadata={"source": str(source_audio) if not isinstance(source_audio, np.ndarray) else "array"},
        )
        
        # Step 1: Load source audio
        logger.info("Loading source audio...")
        if isinstance(source_audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(source_audio))
        else:
            audio = source_audio
            sample_rate = cfg.output_sample_rate
            
        result.metadata["input_sample_rate"] = sample_rate
        result.metadata["input_length"] = len(audio) if audio.ndim == 1 else audio.shape[-1]
        
        # Step 2: Separate vocals
        vocals = audio
        accompaniment = None
        
        if cfg.separate_vocals:
            logger.info("Separating vocals...")
            try:
                if cfg.separation_mode == "2stems":
                    vocals, accompaniment = self.separator.separate(
                        audio, 
                        mode="2stems",
                        sample_rate=sample_rate,
                    )
                else:
                    vocals, drums, bass, other = self.separator.separate(
                        audio,
                        mode="4stems",
                        sample_rate=sample_rate,
                    )
                    # Combine non-vocal stems
                    accompaniment = drums + bass + other
                    
                result.separated_vocals = vocals
                result.separated_accompaniment = accompaniment
                logger.info("Vocal separation complete")
            except Exception as e:
                logger.warning(f"Vocal separation failed: {e}. Using original audio.")
                vocals = audio
                accompaniment = None
        
        # Step 3: Dereverberate
        if cfg.dereverb:
            logger.info("Dereverberating vocals...")
            try:
                vocals = self.separator.dereverb(
                    vocals,
                    sample_rate=sample_rate,
                    method=cfg.dereverb_method,
                    reduction_db=cfg.dereverb_reduction_db,
                )
                result.dereverberated_vocals = vocals
                logger.info("Dereverberation complete")
            except Exception as e:
                logger.warning(f"Dereverberation failed: {e}")
        
        # Step 4: Voice conversion
        logger.info("Converting voice...")
        try:
            converted = self.inference.convert(
                vocals,
                sample_rate=sample_rate,
                transpose=cfg.transpose,
            )
            result.converted_vocals = converted
            logger.info("Voice conversion complete")
        except Exception as e:
            logger.error(f"Voice conversion failed: {e}")
            raise
        
        # Step 5: Mix with accompaniment
        if cfg.mix_with_accompaniment and accompaniment is not None:
            logger.info("Mixing with accompaniment...")
            # Ensure same length
            min_len = min(len(converted), len(accompaniment))
            if converted.ndim == 1:
                converted = converted[:min_len]
            else:
                converted = converted[..., :min_len]
                
            if accompaniment.ndim == 1:
                accompaniment = accompaniment[:min_len]
            else:
                accompaniment = accompaniment[..., :min_len]
            
            # Mix
            output = converted * cfg.vocal_volume + accompaniment * cfg.accompaniment_volume
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output = output / max_val * 0.95
        else:
            output = converted
            
        result.output_audio = output
        result.sample_rate = sample_rate
        
        # Step 6: Save output
        if output_path is not None:
            logger.info(f"Saving output to {output_path}...")
            self._save_audio(output, str(output_path), sample_rate, cfg.output_format)
            result.output_path = str(output_path)
            logger.info("Output saved successfully")
        
        result.metadata["pipeline_complete"] = True
        return result
    
    def separate_only(
        self,
        audio: Union[str, Path, np.ndarray],
        mode: str = "2stems",
    ) -> Dict[str, np.ndarray]:
        """
        Only perform audio separation without voice conversion.
        
        Args:
            audio: Input audio path or array.
            mode: Separation mode ("2stems" or "4stems").
            
        Returns:
            Dictionary with separated stems.
        """
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio))
        else:
            sample_rate = self.config.output_sample_rate
            
        if mode == "2stems":
            vocals, accompaniment = self.separator.separate(
                audio, mode="2stems", sample_rate=sample_rate
            )
            return {"vocals": vocals, "accompaniment": accompaniment}
        else:
            vocals, drums, bass, other = self.separator.separate(
                audio, mode="4stems", sample_rate=sample_rate
            )
            return {
                "vocals": vocals,
                "drums": drums,
                "bass": bass,
                "other": other,
            }
    
    def convert_only(
        self,
        audio: Union[str, Path, np.ndarray],
        transpose: int = 0,
    ) -> np.ndarray:
        """
        Only perform voice conversion without separation.
        
        Args:
            audio: Input audio path or array.
            transpose: Pitch shift in semitones.
            
        Returns:
            Converted audio array.
        """
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio))
        else:
            sample_rate = self.config.output_sample_rate
            
        return self.inference.convert(
            audio,
            sample_rate=sample_rate,
            transpose=transpose,
        )
    
    def _load_audio(self, path: str) -> tuple:
        """Load audio from file."""
        try:
            import librosa
            audio, sr = librosa.load(path, sr=None, mono=False)
            return audio, sr
        except ImportError:
            from scipy.io import wavfile
            sr, audio = wavfile.read(path)
            audio = audio.astype(np.float32) / 32768.0
            return audio, sr
    
    def _save_audio(
        self,
        audio: np.ndarray,
        path: str,
        sample_rate: int,
        format: str = "wav",
    ):
        """Save audio to file."""
        # Ensure output directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to 16-bit PCM
        if audio.dtype != np.int16:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        
        if format == "wav":
            from scipy.io import wavfile
            wavfile.write(path, sample_rate, audio)
        elif format == "mp3":
            try:
                import soundfile as sf
                sf.write(path, audio, sample_rate, format='MP3')
            except ImportError:
                logger.warning("soundfile not available, saving as WAV instead")
                path = path.rsplit('.', 1)[0] + '.wav'
                from scipy.io import wavfile
                wavfile.write(path, sample_rate, audio)
        elif format == "flac":
            try:
                import soundfile as sf
                sf.write(path, audio, sample_rate, format='FLAC')
            except ImportError:
                logger.warning("soundfile not available, saving as WAV instead")
                path = path.rsplit('.', 1)[0] + '.wav'
                from scipy.io import wavfile
                wavfile.write(path, sample_rate, audio)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline information.
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "config": {
                "separate_vocals": self.config.separate_vocals,
                "separation_mode": self.config.separation_mode,
                "separation_backend": self.config.separation_backend,
                "dereverb": self.config.dereverb,
                "transpose": self.config.transpose,
                "mix_with_accompaniment": self.config.mix_with_accompaniment,
            },
            "available_backends": self.separator.get_available_backends() if self._separator else [],
        }
