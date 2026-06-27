"""
Unit tests for audio validation utilities.
"""

import pytest
import numpy as np
from src.utils.audio.validation import (
    validate_audio_input,
    ensure_channel_first,
    ensure_mono,
)


class TestValidateAudioInput:
    """Tests for validate_audio_input function."""
    
    def test_valid_mono_audio(self):
        """Test valid mono audio input."""
        audio = np.random.randn(16000).astype(np.float32)
        result = validate_audio_input(audio)
        assert result.shape == (16000,)
        assert result.dtype == np.float32
    
    def test_valid_stereo_audio_channel_first(self):
        """Test valid stereo audio in channel-first format."""
        audio = np.random.randn(2, 16000).astype(np.float32)
        result = validate_audio_input(audio)
        assert result.shape == (2, 16000)
        assert result.dtype == np.float32
    
    def test_valid_stereo_audio_channel_last(self):
        """Test valid stereo audio in channel-last format (should be transposed)."""
        audio = np.random.randn(16000, 2).astype(np.float32)
        result = validate_audio_input(audio)
        assert result.shape == (2, 16000)
        assert result.dtype == np.float32
    
    def test_empty_audio_raises_error(self):
        """Test that empty audio raises ValueError."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="Audio too short"):
            validate_audio_input(audio, min_samples=1)
    
    def test_zero_dimensional_audio_raises_error(self):
        """Test that 0-dimensional audio raises ValueError."""
        audio = np.array(0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="0-dimensional"):
            validate_audio_input(audio)
    
    def test_3d_audio_raises_error(self):
        """Test that 3D audio raises ValueError."""
        audio = np.random.randn(2, 16000, 2).astype(np.float32)
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            validate_audio_input(audio)
    
    def test_expected_channels_mismatch(self):
        """Test that channel mismatch raises ValueError."""
        audio = np.random.randn(2, 16000).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 1 channels"):
            validate_audio_input(audio, expected_channels=1)
    
    def test_min_samples_check(self):
        """Test that min_samples check works."""
        audio = np.random.randn(100).astype(np.float32)
        with pytest.raises(ValueError, match="Audio too short"):
            validate_audio_input(audio, min_samples=1000)
    
    def test_non_numpy_input_converted(self):
        """Test that non-numpy input is converted to numpy array."""
        audio = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = validate_audio_input(audio)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
    
    def test_dtype_conversion(self):
        """Test that non-float32 dtype is converted."""
        audio = np.random.randn(16000).astype(np.float64)
        result = validate_audio_input(audio)
        assert result.dtype == np.float32
    
    def test_int_dtype_conversion(self):
        """Test that integer dtype is converted to float32."""
        audio = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        result = validate_audio_input(audio)
        assert result.dtype == np.float32
    
    def test_multichannel_audio(self):
        """Test multi-channel audio (5.1 surround)."""
        audio = np.random.randn(6, 16000).astype(np.float32)
        result = validate_audio_input(audio)
        assert result.shape == (6, 16000)
    
    def test_large_audio(self):
        """Test large audio array (1 minute at 44.1kHz stereo)."""
        audio = np.random.randn(2, 44100 * 60).astype(np.float32)
        result = validate_audio_input(audio)
        assert result.shape == (2, 44100 * 60)


class TestEnsureChannelFirst:
    """Tests for ensure_channel_first function."""
    
    def test_mono_passthrough(self):
        """Test that mono audio passes through unchanged."""
        audio = np.random.randn(16000).astype(np.float32)
        result = ensure_channel_first(audio)
        assert result.shape == (16000,)
    
    def test_channel_first_passthrough(self):
        """Test that channel-first audio passes through unchanged."""
        audio = np.random.randn(2, 16000).astype(np.float32)
        result = ensure_channel_first(audio)
        assert result.shape == (2, 16000)
    
    def test_channel_last_transposed(self):
        """Test that channel-last audio is transposed."""
        audio = np.random.randn(16000, 2).astype(np.float32)
        result = ensure_channel_first(audio)
        assert result.shape == (2, 16000)
    
    def test_3d_raises_error(self):
        """Test that 3D audio raises ValueError."""
        audio = np.random.randn(2, 16000, 2).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 1D or 2D"):
            ensure_channel_first(audio)


class TestEnsureMono:
    """Tests for ensure_mono function."""
    
    def test_mono_passthrough(self):
        """Test that mono audio passes through unchanged."""
        audio = np.random.randn(16000).astype(np.float32)
        result = ensure_mono(audio)
        assert result.shape == (16000,)
    
    def test_stereo_to_mono(self):
        """Test that stereo audio is converted to mono."""
        audio = np.random.randn(2, 16000).astype(np.float32)
        result = ensure_mono(audio)
        assert result.shape == (16000,)
    
    def test_multichannel_to_mono(self):
        """Test that multi-channel audio is converted to mono."""
        audio = np.random.randn(6, 16000).astype(np.float32)
        result = ensure_mono(audio)
        assert result.shape == (16000,)
