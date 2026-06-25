"""
Audio Converter - AudioFormat converter
Supports conversion between multiple audio formats
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List
import numpy as np

import ffmpeg


class ConversionFormat(Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    AAC = "aac"
    OGG = "ogg"
    M4A = "m4a"
    WMA = "wma"
    AIFF = "aiff"
    AMR = "amr"
    
    # Lossless formats
    LOSSLESS_FORMATS = {WAV, FLAC, AIFF}
    
    # Lossy formats
    LOSSY_FORMATS = {MP3, AAC, OGG, M4A, WMA, AMR}


@dataclass
class AudioMetadata:
    """Audio metadata"""
    format: str
    sample_rate: int
    channels: int
    bit_rate: Optional[int] = None
    duration: Optional[float] = None
    artist: Optional[str] = None
    title: Optional[str] = None
    album: Optional[str] = None
    year: Optional[int] = None
    genre: Optional[str] = None


class AudioConverter:
    """
    Audio format converter
    
    Implements high-quality audio format conversion based on FFmpeg.
    
    Supports conversion:
    - Format conversion (MP3 -> WAV, FLAC -> MP3, etc.)
    - Sample rate conversion (44100 -> 48000)
    - Channel conversion (Stereo -> Mono)
    - Bitrate adjustment
    - Quality presets
    """
    
    # Quality presets
    QUALITY_PRESETS = {
        "ultra": {"codec": "libFLAC", "compression": 0},
        "high": {"codec": "libFLAC", "compression": 3},
        "medium": {"codec": "libmp3lame", "qscale": 2},
        "low": {"codec": "libmp3lame", "qscale": 4},
    }
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        Initialize converter
        
        Args:
            ffmpeg_path: FFmpeg executable file path
        """
        self.ffmpeg_path = ffmpeg_path
    
    def convert(
        self,
        input_path: str,
        output_path: str,
        output_format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        bit_rate: Optional[str] = None,
        quality: str = "high",
        **kwargs
    ) -> bool:
        """
        Convert audio file
        
        Args:
            input_path: Input file path
            output_path: Output file path
            output_format: Output format (wav, mp3, flac, etc.)
            sample_rate: Target sample rate
            channels: Target channel count
            bit_rate: Target bit rate
            quality: Quality preset
            **kwargs: Other FFmpeg parameters
            
        Returns:
            bool: Whether conversion succeeded
        """
        # Determine output format
        if output_format is None:
            output_format = Path(output_path).suffix[1:].lower()
        
        try:
            # Build FFmpeg command
            stream = ffmpeg.input(input_path)
            
            # Audio filters
            filters = []
            if sample_rate:
                filters.append(f"aformat=sample_fmts=fltp:sample_rates={sample_rate}")
            if channels:
                if channels == 1:
                    filters.append("aformat=channel_layouts=mono")
                elif channels == 2:
                    filters.append("aformat=channel_layouts=stereo")
            
            if filters:
                stream = ffmpeg.filter(stream, 'afilter', ','.join(filters))
            
            # Get encoder settings
            codec_settings = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS["high"])
            
            # Build output parameters
            output_kwargs = {}
            if "codec" in codec_settings:
                output_kwargs["acodec"] = codec_settings["codec"]
            if "qscale" in codec_settings:
                output_kwargs["aq"] = codec_settings["qscale"]
            if bit_rate:
                output_kwargs["audio_bitrate"] = bit_rate
            
            # Add extra parameters
            output_kwargs.update(kwargs)
            
            # Execute conversion
            ffmpeg.output(stream, output_path, **output_kwargs).run(
                cmd=self.ffmpeg_path,
                overwrite_output=True,
                quiet=True,
            )
            
            return True
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return False
    
    def convert_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        output_format: str = "wav",
        **kwargs
    ) -> bool:
        """
        Convert audio array to file
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            output_path: Output file path
            output_format: Output format
            **kwargs: Other parameters
            
        Returns:
            bool: Whether conversion succeeded
        """
        try:
            import soundfile as sf
            
            # Ensure audio format is correct
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            elif audio.shape[0] > audio.shape[1]:
                audio = audio.T
            
            # WriteFile
            sf.write(output_path, audio.T, sample_rate, format=output_format.upper())
            return True
            
        except ImportError:
            print("soundfile not installed. Use convert() for file conversion.")
            return False
        except Exception as e:
            print(f"Error writing audio: {e}")
            return False
    
    def get_metadata(self, file_path: str) -> Optional[AudioMetadata]:
        """
        Get audio metadata
        
        Args:
            file_path: Audio file path
            
        Returns:
            AudioMetadata: Metadata object
        """
        try:
            probe = ffmpeg.probe(file_path, cmd=self.ffmpeg_path)
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"),
                None
            )
            
            if audio_stream is None:
                return None
            
            format_info = probe["format"]
            
            # Parse tags
            tags = audio_stream.get("tags", {})
            
            return AudioMetadata(
                format=format_info.get("format_name", "unknown"),
                sample_rate=int(audio_stream.get("sample_rate", 44100)),
                channels=int(audio_stream.get("channels", 2)),
                bit_rate=int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None,
                duration=float(format_info.get("duration", 0)),
                artist=tags.get("artist"),
                title=tags.get("title"),
                album=tags.get("album"),
                year=int(tags.get("date")) if tags.get("date") else None,
                genre=tags.get("genre"),
            )
            
        except ffmpeg.Error:
            return None
    
    def batch_convert(
        self,
        input_files: List[str],
        output_dir: str,
        output_format: str,
        **kwargs
    ) -> dict:
        """
        Batch convert
        
        Args:
            input_files: Input file list
            output_dir: Output directory
            output_format: Output format
            **kwargs: Conversion parameters
            
        Returns:
            dict: Conversion results {file: success}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for input_file in input_files:
            input_name = Path(input_file).stem
            output_file = str(output_path / f"{input_name}.{output_format}")
            results[input_file] = self.convert(input_file, output_file, output_format, **kwargs)
        
        return results
    
    def normalize_audio(
        self,
        input_path: str,
        output_path: str,
        target_db: float = -20.0,
    ) -> bool:
        """
        Normalize audio level
        
        Args:
            input_path: Input file
            output_path: Output file
            target_db: Target decibel value
            
        Returns:
            bool: Whether succeeded
        """
        try:
            stream = ffmpeg.input(input_path)
            
            # loudnorm Filter
            filtered = ffmpeg.filter(
                stream,
                "loudnorm",
                I=str(target_db),
                TP=-1.5,
                LRA=11,
            )
            
            ffmpeg.output(filtered, output_path, overwrite_output=True).run(
                cmd=self.ffmpeg_path,
                quiet=True,
            )
            return True
            
        except ffmpeg.Error:
            return False
    
    def trim_audio(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: Optional[float] = None,
    ) -> bool:
        """
        Trim audio
        
        Args:
            input_path: Input file
            output_path: Output file
            start_time: Start time (seconds)
            end_time: End time (seconds)
            
        Returns:
            bool: Whether succeeded
        """
        try:
            if end_time:
                stream = ffmpeg.input(input_path, ss=start_time, to=end_time)
            else:
                stream = ffmpeg.input(input_path, ss=start_time)
            
            ffmpeg.output(stream, output_path, overwrite_output=True).run(
                cmd=self.ffmpeg_path,
                quiet=True,
            )
            return True
            
        except ffmpeg.Error:
            return False
    
    def merge_audio(
        self,
        input_files: List[str],
        output_path: str,
        crossfade: float = 0.0,
    ) -> bool:
        """
        Merge multiple audio files
        
        Args:
            input_files: Input file list
            output_path: Output file
            crossfade: Crossfade duration
            
        Returns:
            bool: Whether succeeded
        """
        if not input_files:
            return False
        
        try:
            if len(input_files) == 1:
                # Single file, just copy
                import shutil
                shutil.copy(input_files[0], output_path)
                return True
            
            # Complex merge uses filter_complex
            inputs = [ffmpeg.input(f) for f in input_files]
            
            if crossfade > 0:
                # Merge with crossfade
                merged = inputs[0]
                for inp in inputs[1:]:
                    merged = ffmpeg.filter(
                        [merged, inp],
                        "acrossfade",
                        d=crossfade,
                    )
            else:
                # Direct concatenation
                merged = ffmpeg.concat(*inputs, a=1)
            
            ffmpeg.output(merged, output_path, overwrite_output=True).run(
                cmd=self.ffmpeg_path,
                quiet=True,
            )
            return True
            
        except ffmpeg.Error:
            return False
