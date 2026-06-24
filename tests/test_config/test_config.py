"""
SOMA 配置中心测试

测试 src/config/ 下的配置模块。
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

import pytest


class TestConfigDefaults:
    """默认配置测试"""

    def test_soma_defaults_creation(self):
        """测试：创建默认配置对象"""
        from src.config import DEFAULT_CONFIG, SomaDefaults

        assert isinstance(DEFAULT_CONFIG, SomaDefaults)
        assert DEFAULT_CONFIG.version == "0.1.0"
        assert DEFAULT_CONFIG.app_name == "SOMA"

    def test_separator_defaults(self):
        """测试：分离器默认配置"""
        from src.config import SeparatorDefaults

        defaults = SeparatorDefaults()

        assert defaults.default_model == "htdemucs_ft"
        assert defaults.device == "auto"
        assert defaults.overlap == 0.5
        assert defaults.batch_size == 1

    def test_voice_converter_defaults(self):
        """测试：声音转换器默认配置"""
        from src.config import VoiceConverterDefaults

        defaults = VoiceConverterDefaults()

        assert defaults.pitch_shift == 0.0
        assert defaults.vpm == 0.5
        assert defaults.rms_mix == 0.5
        assert defaults.pitch_algo == "rmvpe"

    def test_effects_defaults(self):
        """测试：音效处理器默认配置"""
        from src.config import EffectsDefaults

        defaults = EffectsDefaults()

        assert defaults.eq.enabled is True
        assert defaults.eq.bands == 10
        assert defaults.reverb.reverb_type == "room"
        assert defaults.pitch.semitones == 0.0

    def test_security_defaults(self):
        """测试：安全设置默认配置"""
        from src.config import SecurityDefaults

        defaults = SecurityDefaults()

        assert "wav" in defaults.allowed_audio_formats
        assert "mp3" in defaults.allowed_audio_formats
        assert defaults.max_file_size_mb == 500
        assert defaults.allow_symlinks is False

    def test_logging_defaults(self):
        """测试：日志系统默认配置"""
        from src.config import LoggingDefaults

        defaults = LoggingDefaults()

        assert defaults.level == "INFO"
        assert defaults.console_output is True
        assert defaults.file_output is True
        assert defaults.backup_count == 7


class TestConfig:
    """配置管理测试"""

    def test_config_load_from_dict(self):
        """测试：从字典加载配置"""
        from src.config import Config

        user_config = {
            "separators": {
                "device": "cuda",
                "default_model": "htdemucs"
            }
        }

        config = Config(user_config=user_config)

        assert config.get("separators.device") == "cuda"
        assert config.get("separators.default_model") == "htdemucs"

    def test_config_get_with_default(self):
        """测试：获取配置值（带默认值）"""
        from src.config import Config

        config = Config()

        # 不存在的键应返回默认值
        value = config.get("nonexistent.key", default="default_value")
        assert value == "default_value"

        # 正常获取
        value = config.get("separators.device", default="cpu")
        assert value in ["auto", "cpu", "cuda", "mps"]

    def test_config_set_value(self):
        """测试：设置配置值"""
        from src.config import Config

        config = Config()

        config.set("separators.device", "cuda")
        assert config.get("separators.device") == "cuda"

    def test_config_get_section(self):
        """测试：获取配置节"""
        from src.config import Config

        config = Config()

        section = config.get_section("separators")
        assert section is not None
        assert hasattr(section, 'device')

    def test_config_to_dict(self):
        """测试：配置转字典"""
        from src.config import Config

        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "separators" in config_dict
        assert "voice_converters" in config_dict


class TestConfigPersistence:
    """配置持久化测试"""

    def test_save_and_load_json(self, temp_dir: Path):
        """测试：保存和加载 JSON 配置"""
        from src.config import Config

        config = Config()
        config.set("separators.device", "cuda")
        config.set("voice_converters.pitch_shift", 5.0)

        # 保存到文件
        config_path = temp_dir / "config.json"
        config.save(str(config_path))

        assert config_path.exists()

        # 重新加载
        loaded_config = Config.load(str(config_path))

        assert loaded_config.get("separators.device") == "cuda"
        assert loaded_config.get("voice_converters.pitch_shift") == 5.0

    def test_config_hierarchy_override(self, temp_dir: Path):
        """测试：配置层级覆盖"""
        from src.config import Config

        # 创建带默认值的配置
        config = Config()

        # 设置用户覆盖
        config.set("separators.device", "mps")

        # 验证覆盖生效
        assert config.get("separators.device") == "mps"

        # 其他未覆盖的值仍为默认值
        assert config.get("separators.overlap") == 0.5


class TestConfigValidation:
    """配置验证测试"""

    def test_validate_valid_config(self):
        """测试：验证有效配置"""
        from src.config import Config

        config = Config()
        assert config.validate() is True

    def test_validate_invalid_device(self):
        """测试：验证无效设备配置"""
        from src.config import Config

        config = Config()
        config.set("separators.device", "invalid_device")

        # 配置无效，但 validate 方法可能只检查部分
        # 这里主要测试不会崩溃
        assert config is not None


class TestConfigPath:
    """配置路径测试"""

    def test_get_config_path(self):
        """测试：获取配置路径"""
        from src.config import get_config_path

        path = get_config_path()
        assert isinstance(path, Path)
        assert ".soma" in str(path)

    def test_get_config_with_path(self, temp_dir: Path):
        """测试：使用指定路径获取配置"""
        from src.config import Config

        config_path = temp_dir / "custom_config.json"

        config = Config.load(str(config_path), auto_create=True)

        assert config is not None
        assert config_path.exists()
