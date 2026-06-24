"""
SOMA 测试框架 - pytest 配置

提供测试所需的共享 fixtures 和配置。
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Optional

import pytest
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """获取项目根目录"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """获取测试数据目录"""
    return project_root / "tests" / "data"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录，测试后自动清理"""
    temp = Path(tempfile.mkdtemp())
    yield temp
    if temp.exists():
        shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def temp_audio_file(temp_dir: Path) -> Generator[Path, None, None]:
    """创建临时音频文件（WAV格式）"""
    audio_path = temp_dir / "test_audio.wav"

    # 生成测试音频数据
    sample_rate = 44100
    duration = 1.0  # 1秒
    samples = int(sample_rate * duration)

    # 生成正弦波
    frequency = 440  # A4
    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5

    # 保存为 WAV 文件
    try:
        import soundfile as sf
        sf.write(str(audio_path), audio_data, sample_rate)
    except ImportError:
        # 如果 soundfile 不可用，使用 scipy
        from scipy.io import wavfile
        # 转换为 int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(str(audio_path), sample_rate, audio_int16)

    yield audio_path

    # 清理
    if audio_path.exists():
        audio_path.unlink()


@pytest.fixture
def temp_model_file(temp_dir: Path) -> Generator[Path, None, None]:
    """创建临时模型文件（模拟 .pth 文件）"""
    model_path = temp_dir / "test_model.pth"

    # 创建一个简单的状态字典
    state_dict = {
        'layer1.weight': np.random.randn(10, 5).astype(np.float32),
        'layer1.bias': np.random.randn(10).astype(np.float32),
    }

    # 保存为 pickle 文件
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(state_dict, f)

    yield model_path

    # 清理
    if model_path.exists():
        model_path.unlink()


@pytest.fixture
def mock_config(temp_dir: Path) -> dict:
    """创建测试用配置"""
    return {
        "soma": {
            "version": "0.1.0",
            "app_dir": str(temp_dir / ".soma"),
            "app_name": "SOMA",
        },
        "separators": {
            "device": "cpu",
            "default_model": "htdemucs_ft",
            "output_format": "wav",
        },
        "voice_converters": {
            "device": "cpu",
            "pitch_shift": 0.0,
            "vpm": 0.5,
            "rms_mix": 0.5,
        },
        "audio_utils": {
            "default_sample_rate": 44100,
            "default_channels": 2,
            "max_file_size_mb": 100,
            "max_duration_seconds": 600,
        },
        "security": {
            "allowed_base_dirs": [str(temp_dir)],
            "allowed_audio_formats": ["wav", "mp3", "flac"],
            "max_file_size_mb": 100,
        },
        "logging": {
            "level": "DEBUG",
            "console_output": False,
            "file_output": False,
        }
    }


@pytest.fixture
def sample_audio_metadata() -> dict:
    """示例音频元数据"""
    return {
        "sample_rate": 44100,
        "channels": 2,
        "duration": 3.0,
        "bit_depth": 16,
        "format": "WAV",
        "codec": "pcm_s16le",
    }


@pytest.fixture
def sample_audio_data() -> np.ndarray:
    """生成示例音频数据"""
    sample_rate = 44100
    duration = 0.1  # 100ms
    samples = int(sample_rate * duration)

    # 生成包含多个频率的测试信号
    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +   # 440Hz
        0.3 * np.sin(2 * np.pi * 880 * t) +   # 880Hz
        0.2 * np.sin(2 * np.pi * 1760 * t)    # 1760Hz
    )
    return audio


# pytest 配置
def pytest_configure(config):
    """pytest 初始化配置"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集行为"""
    # 可以在这里添加自动跳过等逻辑
    pass
