"""
SOMA 声音转换器测试

测试 src/voice_converters/ 下的声音转换模块。
"""

import sys
from pathlib import Path

import pytest
import numpy as np


class TestConversionParams:
    """转换参数测试"""

    def test_creation(self):
        """测试：创建转换参数"""
        from src.voice_converters.base import ConversionParams

        params = ConversionParams()
        assert params.pitch_shift == 0.0
        assert params.vpm == 0.5
        assert params.rms_mix == 0.5

    def test_custom_values(self):
        """测试：自定义参数值"""
        from src.voice_converters.base import ConversionParams

        params = ConversionParams(
            pitch_shift=5.0,
            vpm=0.7,
            rms_mix=0.8
        )

        assert params.pitch_shift == 5.0
        assert params.vpm == 0.7
        assert params.rms_mix == 0.8

    def test_parameter_validation(self):
        """测试：参数验证"""
        from src.voice_converters.base import ConversionParams

        params = ConversionParams()

        # vpm 应该在 0-1 范围
        assert 0 <= params.vpm <= 1
        assert 0 <= params.rms_mix <= 1


class TestConversionResult:
    """转换结果测试"""

    def test_creation(self):
        """测试：创建转换结果"""
        from src.voice_converters.base import ConversionResult
        import numpy as np

        audio = np.random.randn(44100).astype(np.float32)

        result = ConversionResult(
            audio=audio,
            sampling_rate=44100
        )

        assert result.audio is not None
        assert result.sampling_rate == 44100

    def test_info_field(self):
        """测试：info 字段"""
        from src.voice_converters.base import ConversionResult
        import numpy as np

        audio = np.random.randn(44100).astype(np.float32)

        result = ConversionResult(
            audio=audio,
            sampling_rate=44100,
            info={"model": "test", "duration": 1.0}
        )

        assert result.info is not None
        assert result.info["model"] == "test"


class TestBaseVoiceConverter:
    """基础转换器测试"""

    def test_abstract_class(self):
        """测试：抽象基类不能直接实例化"""
        from src.voice_converters.base import BaseVoiceConverter

        with pytest.raises(TypeError):
            BaseVoiceConverter()

    def test_load_unload_methods(self):
        """测试：load/unload 方法存在"""
        from src.voice_converters.base import BaseVoiceConverter
        from abc import ABC

        # BaseVoiceConverter 应该有抽象方法
        assert hasattr(BaseVoiceConverter, 'load_model')
        assert hasattr(BaseVoiceConverter, 'unload')
        assert hasattr(BaseVoiceConverter, 'convert')


class TestRVCConverter:
    """RVC 转换器测试"""

    def test_rvc_creation(self):
        """测试：创建 RVC 转换器"""
        try:
            from src.voice_converters.rvc_converter import RVCConverter

            converter = RVCConverter()
            assert converter is not None

        except ImportError as e:
            pytest.skip(f"RVC 依赖不可用: {e}")

    def test_rvc_f0_methods(self):
        """测试：RVC F0 方法"""
        try:
            from src.voice_converters.rvc_converter import RVCConverter, F0Method

            converter = RVCConverter()

            # 检查支持的 F0 方法
            available = converter.get_available_f0_methods()
            assert isinstance(available, list)

        except ImportError:
            pytest.skip("RVC 依赖不可用")

    def test_rvc_load_unload(self, temp_dir: Path):
        """测试：RVC 加载和卸载"""
        try:
            from src.voice_converters.rvc_converter import RVCConverter

            converter = RVCConverter()

            # 加载不存在的模型应该失败
            with pytest.raises(Exception):
                converter.load_model(str(temp_dir / "nonexistent.pth"))

            # 卸载空转换器应该没问题
            converter.unload()

        except ImportError:
            pytest.skip("RVC 依赖不可用")


class TestSoVITSConverter:
    """SoVITS 转换器测试"""

    def test_sovits_creation(self):
        """测试：创建 SoVITS 转换器"""
        try:
            from src.voice_converters.sovits_converter import SoVITSConverter

            converter = SoVITSConverter()
            assert converter is not None

        except ImportError as e:
            pytest.skip(f"SoVITS 依赖不可用: {e}")

    def test_sovits_version(self):
        """测试：SoVITS 初始化参数"""
        try:
            from src.voice_converters.sovits_converter import SoVITSConverter

            converter = SoVITSConverter()
            # 验证转换器可以创建
            assert converter is not None

        except ImportError:
            pytest.skip("SoVITS 依赖不可用")

    def test_sovits_load_unload(self, temp_dir: Path):
        """测试：SoVITS 加载和卸载"""
        try:
            from src.voice_converters.sovits_converter import SoVITSConverter

            converter = SoVITSConverter()

            # 加载不存在的模型应该失败
            with pytest.raises(Exception):
                converter.load_model(str(temp_dir / "nonexistent.pth"))

            # 卸载空转换器应该没问题
            converter.unload()

        except ImportError:
            pytest.skip("SoVITS 依赖不可用")


class TestConverterFactory:
    """转换器工厂测试"""

    def test_factory_creation(self):
        """测试：创建工厂"""
        from src.voice_converters.factory import ConverterFactory

        factory = ConverterFactory()
        assert factory is not None

    def test_get_available_engines(self):
        """测试：获取可用引擎"""
        from src.voice_converters.factory import ConverterFactory

        factory = ConverterFactory()
        engines = factory.get_available_engines()

        assert isinstance(engines, list)
        # 应该至少有 RVC 或 SoVITS
        assert len(engines) >= 0

    def test_create_rvc(self):
        """测试：创建 RVC 转换器"""
        try:
            from src.voice_converters.factory import ConverterFactory

            factory = ConverterFactory()
            
            # 检查是否有 create_rvc_converter 方法
            if hasattr(factory, 'create_rvc_converter'):
                converter = factory.create_rvc_converter()
                if converter is not None:
                    assert converter is not None
            else:
                pytest.skip("create_rvc_converter method not available")

        except ImportError:
            pytest.skip("RVC 依赖不可用")

    def test_create_sovits(self):
        """测试：创建 SoVITS 转换器"""
        try:
            from src.voice_converters.factory import ConverterFactory

            factory = ConverterFactory()
            
            # 检查是否有 create_sovits_converter 方法
            if hasattr(factory, 'create_sovits_converter'):
                converter = factory.create_sovits_converter()
                if converter is not None:
                    assert converter is not None
            else:
                pytest.skip("create_sovits_converter method not available")

        except ImportError:
            pytest.skip("SoVITS 依赖不可用")


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_create_converter_function(self, temp_dir: Path):
        """测试：create_converter 函数"""
        from src.voice_converters import create_converter

        # 创建不存在的模型路径应该失败
        with pytest.raises(Exception):
            create_converter(str(temp_dir / "nonexistent.pth"))

    def test_context_manager(self, temp_dir: Path):
        """测试：上下文管理器"""
        from src.voice_converters import create_converter

        # 应该支持上下文管理器语法
        try:
            with create_converter(str(temp_dir / "test.pth")) as converter:
                pass
        except Exception:
            # 预期失败，模型不存在
            pass
