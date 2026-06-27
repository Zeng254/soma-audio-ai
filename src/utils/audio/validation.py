"""
Common audio validation utilities.

This module provides shared audio validation functions used across
separators, effects, and other audio processing components.
"""

import numpy as np
from typing import Optional


def validate_audio_input(
    audio: np.ndarray,
    expected_channels: Optional[int] = None,
    min_samples: int = 1,
) -> np.ndarray:
    """
    Validate and normalize audio input to channel-first format.
    
    This function ensures audio data is in the expected format:
    - dtype: float32
    - shape: (channels, samples) for stereo/multi-channel, or (samples,) for mono
    
    Args:
        audio: Input audio array
        expected_channels: Expected number of channels (None for any)
        min_samples: Minimum number of samples required
        
    Returns:
        Validated audio array in channel-first format
        
    Raises:
        ValueError: If audio is invalid or doesn't meet requirements
    """
    # Convert to numpy array if not already
    if not isinstance(audio, np.ndarray):
        try:
            audio = np.array(audio, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot convert audio to numpy array: {e}")
    
    # Ensure float32 dtype
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Handle different dimensionalities
    if audio.ndim == 0:
        raise ValueError("Audio array cannot be 0-dimensional (scalar)")
    
    elif audio.ndim == 1:
        # Mono audio - keep as 1D
        if len(audio) < min_samples:
            raise ValueError(
                f"Audio too short: {len(audio)} samples, minimum is {min_samples}"
            )
        return audio
    
    elif audio.ndim == 2:
        # 2D audio - determine if channel-first or channel-last
        dim0, dim1 = audio.shape
        
        # Heuristic: if one dimension is small (<=8), it's likely channels
        # Audio typically has at most 8 channels (7.1 surround)
        if dim0 <= 8 and dim1 > dim0 * 2:
            # Already channel-first (channels, samples)
            if dim1 < min_samples:
                raise ValueError(
                    f"Audio too short: {dim1} samples, minimum is {min_samples}"
                )
            if expected_channels is not None and dim0 != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} channels, got {dim0}"
                )
            return audio
        
        elif dim1 <= 8 and dim0 > dim1 * 2:
            # Channel-last (samples, channels) - transpose to channel-first
            audio = audio.T
            if dim0 < min_samples:
                raise ValueError(
                    f"Audio too short: {dim0} samples, minimum is {min_samples}"
                )
            if expected_channels is not None and dim1 != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} channels, got {dim1}"
                )
            return audio
        
        else:
            # Ambiguous - assume channel-first
            if dim1 < min_samples:
                raise ValueError(
                    f"Audio too short: {dim1} samples, minimum is {min_samples}"
                )
            return audio
    
    else:
        raise ValueError(
            f"Audio array must be 1D or 2D, got {audio.ndim}D with shape {audio.shape}"
        )


def ensure_channel_first(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is in channel-first format (channels, samples).
    
    Args:
        audio: Input audio array
        
    Returns:
        Audio in channel-first format
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        dim0, dim1 = audio.shape
        if dim0 <= 8 and dim1 > dim0 * 2:
            return audio  # Already channel-first
        elif dim1 <= 8 and dim0 > dim1 * 2:
            return audio.T  # Transpose from channel-last
        else:
            return audio  # Assume channel-first
    else:
        raise ValueError(f"Expected 1D or 2D audio, got {audio.ndim}D")


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono by averaging channels.
    
    Args:
        audio: Input audio array in channel-first format
        
    Returns:
        Mono audio array (1D)
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        return np.mean(audio, axis=0)
    else:
        raise ValueError(f"Expected 1D or 2D audio, got {audio.ndim}D")


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to stereo by duplicating mono or keeping existing stereo.
    
    Args:
        audio: Input audio array
        
    Returns:
        Stereo audio array in channel-first format (2, samples)
    """
    audio = ensure_channel_first(audio)
    
    if audio.ndim == 1:
        # Mono to stereo
        return np.stack([audio, audio], axis=0)
    elif audio.ndim == 2:
        if audio.shape[0] == 1:
            # Mono (1, samples) to stereo
            return np.concatenate([audio, audio], axis=0)
        elif audio.shape[0] == 2:
            # Already stereo
            return audio
        else:
            # Multi-channel to stereo (take first two or downmix)
            if audio.shape[0] > 2:
                return audio[:2]
            return audio
    else:
        raise ValueError(f"Expected 1D or 2D audio, got {audio.ndim}D")


__all__ = [
    "validate_audio_input",
    "ensure_channel_first",
    "ensure_mono",
    "ensure_stereo",
]
