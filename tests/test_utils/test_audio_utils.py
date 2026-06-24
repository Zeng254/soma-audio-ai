"""
SOMA 工具模块测试

测试 src/utils/ 下的工具模块：
- audio_io 音频读写工具
- validator 参数校验工具
- logger 日志系统
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

import pytest
import numpy as np

# 导入待测试模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestAudioIO:
    """音频读写工具测试"""

    def test_load_audio_basic(self, temp_audio_file: Path, sample_audio_data: np.ndarray):
        """测试：加载音频文件基本功能"""
        try:
            from src.utils.audio_io import AudioReader

            reader = AudioReader()
            audio, sr = reader.load(str(temp_audio_file))

            assert sr == 44100, f"采样率应为 44100，实际为 {sr}"
            assert audio is not None, "音频数据不应为空"
            assert len(audio) > 0, "音频长度应大于 0"

        except ImportError as e:
            pytest.skip(f"缺少依赖: {e}")

    def test_load_audio_with_resample(self, temp_dir: Path, sample_audio_data: np.ndarray):
        """测试：带重采样的音频加载"""
        try:
            from src.utils.audio_io import AudioReader

            # 创建测试文件
            test_file = temp_dir / "test_resample.wav"
            try:
                import soundfile as sf
                sf.write(str(test_file), sample_audio_data, 44100)
            except ImportError:
                from scipy.io import wavfile
                audio_int16 = (sample_audio_data * 32767).astype(np.int16)
                wavfile.write(str(test_file), 44100, audio_int16)

            reader = AudioReader()
            audio, sr = reader.load(str(test_file), target_sr=22050)

            assert sr == 22050, f"重采样后应为 22050，实际为 {sr}"

        except ImportError as e:
            pytest.skip(f"缺少依赖: {e}")

    def test_save_audio(self, temp_dir: Path, sample_audio_data: np.ndarray):
        """测试：保存音频文件"""
        try:
            from src.utils.audio_io import AudioWriter

            output_path = temp_dir / "output.wav"

            writer = AudioWriter()
            writer.save(
                audio=sample_audio_data,
                path=str(output_path),
                sample_rate=44100
            )

            assert output_path.exists(), "输出文件应存在"
            assert output_path.stat().st_size > 0, "文件大小应大于 0"

        except ImportError as e:
            pytest.skip(f"缺少依赖: {e}")

    def test_audio_normalization(self, temp_dir: Path, sample_audio_data: np.ndarray):
        """测试：音频归一化"""
        try:
            from src.utils.audio_io import normalize_audio

            # 创建过大的音频
            loud_audio = sample_audio_data * 10

            normalized = normalize_audio(loud_audio)

            assert np.max(np.abs(normalized)) <= 1.0, "归一化后最大值应 <= 1.0"

        except ImportError as e:
            pytest.skip(f"缺少依赖: {e}")

    def test_load_nonexistent_file(self):
        """测试：加载不存在的文件应抛出异常"""
        try:
            from src.utils.audio_io import AudioReader

            reader = AudioReader()

            with pytest.raises(FileNotFoundError):
                reader.load("/nonexistent/path/audio.wav")

        except ImportError:
            pytest.skip("依赖不可用")

    def test_stereo_to_mono_conversion(self, temp_dir: Path):
        """测试：立体声到单声道的转换"""
        try:
            from src.utils.audio_io import AudioReader

            # 创建立体声测试文件
            stereo_file = temp_dir / "stereo.wav"
            stereo_data = np.random.randn(2, 44100).astype(np.float32) * 0.5

            try:
                import soundfile as sf
                sf.write(str(stereo_file), stereo_data.T, 44100)
            except ImportError:
                from scipy.io import wavfile
                wavfile.write(str(stereo_file), 44100, stereo_data.T.astype(np.int16))

            reader = AudioReader()
            audio, sr = reader.load(str(stereo_file), mono=True)

            assert audio.ndim == 1 or (audio.ndim == 2 and audio.shape[0] == 1), \
                "应转换为单声道"

        except ImportError as e:
            pytest.skip(f"缺少依赖: {e}")


class TestValidator:
    """参数校验工具测试"""

    def test_validate_sample_rate_valid(self):
        """测试：有效的采样率验证"""
        from src.utils.validator import validate_sample_rate

        assert validate_sample_rate(44100) == 44100
        assert validate_sample_rate(48000) == 48000
        assert validate_sample_rate("44100") == 44100  # 字符串输入

    def test_validate_sample_rate_invalid(self):
        """测试：无效的采样率验证"""
        from src.utils.validator import validate_sample_rate, ValidationError

        with pytest.raises(ValidationError):
            validate_sample_rate(100)  # 太低

        with pytest.raises(ValidationError):
            validate_sample_rate(1000000)  # 太高

        with pytest.raises(ValidationError):
            validate_sample_rate(-44100)  # 负数

    def test_validate_pitch_shift_range(self):
        """测试：音调偏移范围验证"""
        from src.utils.validator import validate_pitch_shift

        # 有效范围
        assert validate_pitch_shift(0) == 0
        assert validate_pitch_shift(12) == 12
        assert validate_pitch_shift(-12) == -12

        # 边界值
        assert validate_pitch_shift(24) == 24
        assert validate_pitch_shift(-24) == -24

    def test_validate_pitch_shift_out_of_range(self):
        """测试：超出范围的音调偏移"""
        from src.utils.validator import validate_pitch_shift, ValidationError

        with pytest.raises(ValidationError):
            validate_pitch_shift(25)  # 超过上限

        with pytest.raises(ValidationError):
            validate_pitch_shift(-25)  # 超过下限

    def test_validate_duration(self):
        """测试：音频时长验证"""
        from src.utils.validator import validate_duration

        assert validate_duration(60.0) == 60.0
        assert validate_duration(0.1) == 0.1

    def test_validate_duration_out_of_range(self):
        """测试：超出范围的时长"""
        from src.utils.validator import validate_duration, ValidationError

        with pytest.raises(ValidationError):
            validate_duration(0)  # 必须大于 0

        with pytest.raises(ValidationError):
            validate_duration(-10)  # 不能为负

    def test_validate_model_path_valid(self):
        """测试：有效的模型路径"""
        from src.utils.validator import validate_model_path

        # 有效的扩展名
        assert validate_model_path("model.pth")
        assert validate_model_path("model.pt")
        assert validate_model_path("model.onnx")

    def test_validate_model_path_invalid_ext(self):
        """测试：无效的模型扩展名"""
        from src.utils.validator import validate_model_path, ValidationError

        with pytest.raises(ValidationError):
            validate_model_path("model.txt")  # 不支持的扩展名

    def test_validate_file_format(self):
        """测试：音频格式验证"""
        from src.utils.validator import validate_audio_format

        valid_formats = ["wav", "mp3", "flac", "ogg", "m4a"]
        for fmt in valid_formats:
            assert validate_audio_format(fmt) == fmt.upper()

    def test_validate_float_in_range(self):
        """测试：浮点数范围验证"""
        from src.utils.validator import validate_float

        assert validate_float(0.5, min_val=0.0, max_val=1.0) == 0.5
        assert validate_float("0.5", min_val=0.0, max_val=1.0) == 0.5

    def test_validate_float_out_of_range(self):
        """测试：浮点数超出范围"""
        from src.utils.validator import validate_float, ValidationError

        with pytest.raises(ValidationError):
            validate_float(1.5, min_val=0.0, max_val=1.0)

        with pytest.raises(ValidationError):
            validate_float(-0.1, min_val=0.0, max_val=1.0)


class TestLogger:
    """日志系统测试"""

    def test_logger_creation(self):
        """测试：创建 logger"""
        from src.utils.logger import get_logger

        logger = get_logger("test.module")

        assert logger is not None
        assert logger.name == "test.module"

    def test_logger_singleton(self):
        """测试：logger 单例"""
        from src.utils.logger import get_logger

        logger1 = get_logger("test.singleton")
        logger2 = get_logger("test.singleton")

        assert logger1 is logger2

    def test_set_module_level(self):
        """测试：设置模块日志级别"""
        from src.utils.logger import set_module_level, get_logger

        # 应该不抛出异常
        set_module_level("test.module", "DEBUG")
        set_module_level("test.module", "INFO")
        set_module_level("test.module", "WARNING")

    def test_setup_logging_with_defaults(self):
        """测试：使用默认配置设置日志"""
        from src.utils.logger import setup_logging, get_log_file_path

        setup_logging(level="DEBUG")

        # 获取日志文件路径
        log_path = get_log_file_path()
        # 不一定返回路径（可能配置为不写文件）
        assert log_path is None or isinstance(log_path, Path)

    def test_log_levels(self):
        """测试：不同的日志级别"""
        from src.utils.logger import get_logger, setup_logging

        setup_logging(level="DEBUG", console_output=False, file_output=False)
        logger = get_logger("test.levels")

        # 应该不抛出异常
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")


class TestFileUtils:
    """文件工具测试"""

    def test_get_file_extension(self):
        """测试：获取文件扩展名"""
        from src.utils.file import get_extension

        assert get_extension("audio.wav") == "wav"
        assert get_extension("audio.WAV") == "wav"
        assert get_extension("audio.tar.gz") == "gz"

    def test_ensure_directory_creation(self, temp_dir: Path):
        """测试：确保目录创建"""
        from src.utils.file import ensure_dir

        new_dir = temp_dir / "subdir" / "nested"
        result = ensure_dir(new_dir)

        assert result.exists()
        assert result.is_dir()

    def test_safe_filename(self):
        """测试：安全的文件名"""
        from src.utils.file import safe_filename

        assert safe_filename("audio file.wav") == "audio_file.wav"
        assert safe_filename("../etc/passwd") == "etc_passwd"
        assert safe_filename("audio" * 100 + ".wav") == "audio" * 50 + ".wav"  # 截断
