"""
SOMA pytest 配置和 fixtures

提供测试所需的共享 fixtures 和配置钩子。
"""

import os
import sys
from pathlib import Path

import pytest


# 确保 src 目录在 Python 路径中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """pytest 初始化钩子"""
    # 注册自定义标记
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: marks tests that require GPU (skip if no GPU available)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "requires_deps: marks tests that require optional dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """修改收集到的测试项"""
    # 如果没有 GPU，自动跳过 requires_gpu 测试
    if not _has_gpu():
        skip_gpu = pytest.mark.skip(reason="需要 GPU")
        for item in items:
            if "requires_gpu" in item.keywords:
                item.add_marker(skip_gpu)


def pytest_report_header(config):
    """添加测试报告头信息"""
    info = [
        "SOMA Test Suite",
        f"Python: {sys.version.split()[0]}",
    ]

    try:
        import torch
        if torch.cuda.is_available():
            info.append(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.get_device_name(0)})")
        else:
            info.append(f"PyTorch: {torch.__version__} (CUDA: 不可用)")
    except ImportError:
        info.append("PyTorch: 未安装")

    try:
        import numpy as np
        info.append(f"NumPy: {np.__version__}")
    except ImportError:
        info.append("NumPy: 未安装")

    return info


def _has_gpu() -> bool:
    """检查是否有 GPU 可用"""
    try:
        import torch
        return torch.cuda.is_available() or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
    except ImportError:
        return False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_audio_dir(tmp_path):
    """创建临时音频目录"""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """创建临时模型目录"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def sample_audio_params():
    """示例音频参数"""
    return {
        "sample_rate": 44100,
        "duration": 1.0,
        "channels": 1,
    }


@pytest.fixture
def mock_audio_data():
    """模拟音频数据"""
    import numpy as np
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def mock_stereo_audio():
    """模拟立体声音频数据"""
    import numpy as np
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    audio = np.stack([left, right], axis=0).astype(np.float32)
    return audio, sample_rate


# ============================================================================
# 兼容性别名 fixtures
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path, temp_audio_dir):
    """临时目录（兼容性别名）"""
    return tmp_path


@pytest.fixture
def temp_config_dir(tmp_path):
    """临时配置目录"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_rate():
    """标准采样率"""
    return 44100


@pytest.fixture
def temp_audio_file(temp_audio_dir):
    """临时音频文件路径（temp_audio_file 的别名）"""
    import numpy as np
    import soundfile as sf
    
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    file_path = temp_audio_dir / "test_audio.wav"
    sf.write(str(file_path), audio, sample_rate)
    return str(file_path)


@pytest.fixture
def sample_audio_data():
    """示例音频数据（返回 (audio, sample_rate) 元组）"""
    import numpy as np
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


# 别名，确保两边都能找到
temp_audio_file_fixture = temp_audio_file


@pytest.fixture
def mock_model_file(temp_model_dir):
    """创建临时模型文件（模拟 pickle）"""
    import torch
    
    file_path = temp_model_dir / "model.pth"
    torch.save({"model": torch.randn(10, 10)}, str(file_path))
    return file_path


# 别名
temp_model_file = mock_model_file
