"""
SOMA 音效处理器测试

测试 src/effects/ 下的音效处理模块。
"""

import sys
from pathlib import Path

import pytest
import numpy as np


class TestEqualizer:
    """均衡器测试"""

    def test_equalizer_creation(self):
        """测试：创建均衡器"""
        try:
            from src.effects.eq import Equalizer

            eq = Equalizer(sample_rate=44100)
            assert eq.sample_rate == 44100
            assert len(eq.bands) == 10

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_equalizer_set_gain(self):
        """测试：设置增益"""
        try:
            from src.effects.eq import Equalizer

            eq = Equalizer(sample_rate=44100)
            eq.set_band(index=0, freq=32, gain=3.0)

            assert eq.bands[0]["gain"] == pytest.approx(3.0, rel=0.1)

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_equalizer_preset(self):
        """测试：预设均衡器"""
        try:
            from src.effects.eq import Equalizer

            eq = Equalizer(sample_rate=44100)
            eq.set_preset("vocal_boost")

            # 预设应该改变增益值
            gains = [b["gain"] for b in eq.bands]
            assert any(g != 0 for g in gains)

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_equalizer_flat_preset(self):
        """测试：平坦预设"""
        try:
            from src.effects.eq import Equalizer

            eq = Equalizer(sample_rate=44100)
            eq.set_preset("flat")

            # 平坦预设所有增益应为 0
            gains = [b["gain"] for b in eq.bands]
            assert all(g == 0.0 for g in gains)

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_equalizer_process(self, sample_audio_data: np.ndarray):
        """测试：处理音频"""
        try:
            from src.effects.eq import Equalizer

            eq = Equalizer(sample_rate=44100)
            audio, sr = sample_audio_data
            result = eq.process(audio)

            assert result is not None

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_equalizer_bypass(self, sample_audio_data: np.ndarray):
        """测试：Bypass 模式"""
        try:
            from src.effects.eq import Equalizer

            eq = Equalizer(sample_rate=44100)
            eq.enabled = False

            audio, sr = sample_audio_data
            # 确保输入是1D
            if audio.ndim > 1:
                audio = audio[0] if audio.shape[0] == 1 else audio[:, 0]
            result = eq.process(audio)

            # Bypass 时输出应与输入相同（允许shape差异）
            np.testing.assert_array_almost_equal(result.audio.flatten(), audio.flatten())

        except ImportError:
            pytest.skip("依赖模块不可用")


class TestReverb:
    """混响效果测试"""

    def test_reverb_creation(self):
        """测试：创建混响器"""
        try:
            from src.effects.reverb import Reverb

            reverb = Reverb(sample_rate=44100)
            assert reverb.sample_rate == 44100
            assert reverb.reverb_type == "room"

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_reverb_set_parameters(self):
        """测试：设置混响参数"""
        try:
            from src.effects.reverb import Reverb

            reverb = Reverb(
                sample_rate=44100,
                room_size=0.7,
                damping=0.5,
                wet_level=0.3
            )

            assert reverb.room_size == 0.7
            assert reverb.damping == 0.5
            assert reverb.wet_level == 0.3

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_reverb_preset(self):
        """测试：混响预设"""
        try:
            from src.effects.reverb import Reverb

            reverb = Reverb(sample_rate=44100, reverb_type="hall")

            assert reverb.reverb_type == "hall"

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_reverb_process(self, sample_audio_data: np.ndarray):
        """测试：处理音频"""
        try:
            from src.effects.reverb import Reverb

            reverb = Reverb(sample_rate=44100)
            audio, sr = sample_audio_data
            result = reverb.process(audio)

            assert result is not None

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_reverb_bypass(self, sample_audio_data: np.ndarray):
        """测试：Bypass 模式"""
        try:
            from src.effects.reverb import Reverb

            reverb = Reverb(sample_rate=44100)
            reverb.enabled = False

            audio, sr = sample_audio_data
            # 确保输入是1D
            if audio.ndim > 1:
                audio = audio[0] if audio.shape[0] == 1 else audio[:, 0]
            result = reverb.process(audio)

            # 验证结果存在
            assert result.audio is not None

        except ImportError:
            pytest.skip("依赖模块不可用")


class TestPitchShifter:
    """音调变换测试"""

    def test_pitch_shifter_creation(self):
        """测试：创建音调变换器"""
        try:
            from src.effects.pitch import PitchShifter

            shifter = PitchShifter(sample_rate=44100)
            assert shifter.sample_rate == 44100

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_pitch_shift_set_semitones(self):
        """测试：设置半音调整"""
        try:
            from src.effects.pitch import PitchShifter

            shifter = PitchShifter(sample_rate=44100, semitones=5)

            assert shifter.semitones == 5

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_pitch_shift_range(self):
        """测试：音调范围"""
        try:
            from src.effects.pitch import PitchShifter

            # 设置有效范围
            shifter = PitchShifter(sample_rate=44100, semitones=12)
            assert shifter.semitones == 12

            shifter = PitchShifter(sample_rate=44100, semitones=-12)
            assert shifter.semitones == -12

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_pitch_shift_process(self, sample_audio_data: np.ndarray):
        """测试：处理音频"""
        try:
            from src.effects.pitch import PitchShifter

            shifter = PitchShifter(sample_rate=44100, semitones=0)

            audio, sr = sample_audio_data
            result = shifter.process(audio)

            assert result is not None

        except ImportError:
            pytest.skip("依赖模块不可用")

    def test_pitch_shift_no_change(self, sample_audio_data: np.ndarray):
        """测试：零变换"""
        try:
            from src.effects.pitch import PitchShifter

            shifter = PitchShifter(sample_rate=44100, semitones=0)

            audio, sr = sample_audio_data
            # 确保输入是1D
            if audio.ndim > 1:
                audio = audio[0] if audio.shape[0] == 1 else audio[:, 0]
            result = shifter.process(audio)

            # 零变换应该接近原始信号
            # 注意：由于算法处理，可能略有差异
            result_audio = result.audio.flatten()
            min_len = min(len(audio), len(result_audio))
            correlation = np.corrcoef(audio.flatten()[:min_len], result_audio[:min_len])[0, 1]
            assert correlation > 0.9

        except ImportError:
            pytest.skip("依赖模块不可用")


class TestEffectsPipeline:
    """效果链测试"""

    def test_chain_creation(self):
        """测试：创建效果链"""
        try:
            from src.effects.effects_chain import EffectsChain

            chain = EffectsChain(sample_rate=44100)
            assert chain.sample_rate == 44100
            assert len(chain.effects) == 0

        except ImportError:
            pytest.skip("EffectsChain 尚未实现")

    def test_add_effect(self):
        """测试：添加效果"""
        try:
            from src.effects.effects_chain import EffectsChain
            from src.effects.eq import Equalizer

            chain = EffectsChain(sample_rate=44100)
            eq = Equalizer(sample_rate=44100)

            chain.add(eq)
            assert len(chain.effects) == 1

        except ImportError:
            pytest.skip("EffectsChain 尚未实现")

    def test_process_chain(self, sample_audio_data: np.ndarray):
        """测试：处理效果链"""
        try:
            from src.effects.effects_chain import EffectsChain
            from src.effects.eq import Equalizer

            chain = EffectsChain(sample_rate=44100)
            chain.add(Equalizer(sample_rate=44100))

            result = chain.process(sample_audio_data)
            assert result is not None

        except ImportError:
            pytest.skip("EffectsChain 尚未实现")
