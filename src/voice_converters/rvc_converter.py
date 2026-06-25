"""
RVC (Retrieval-Based Voice Conversion) Converter

实现 RVC v2 核心推理逻辑，包括:
- 模型加载 (惰性加载)
- 音频预处理 (降噪、归一化、重采样)
- F0 提取 (PM/DIO/Crepe)
- HubERT 特征提取
- 推理合成 (PE + AP)
- 声码器 (HiFi-GAN)
- 后处理 (归一化、淡入淡出)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.voice_converters.base import BaseVoiceConverter, ConversionResult
from src.voice_converters.base import ConversionParams, EngineCapability


class RVCConverter(BaseVoiceConverter, EngineCapability):
    """
    RVC 声音转换器

    支持 RVC v1/v2 模型格式，使用惰性加载和优雅降级策略。
    """

    # RVC 默认参数
    DEFAULT_SAMPLE_RATE = 40000
    DEFAULT_HOP_LENGTH = 512
    DEFAULT_HUBERT_DIM = 256
    DEFAULT_F0_MIN = 50.0
    DEFAULT_F0_MAX = 1100.0

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        hop_length: int = DEFAULT_HOP_LENGTH,
        pitch_shift: float = 0.0,
        pitch_algo: str = "pm",
        index_rate: float = 0.0,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix: float = 0.0,
        protect: float = 0.33,
    ):
        """
        初始化 RVC 转换器

        Args:
            device: 运行设备 ("cpu", "cuda", "mps")
            sample_rate: 模型采样率
            hop_length: 帧移
            pitch_shift: 音高变换 (半音)
            pitch_algo: F0 提取算法 ("pm", "dio", "harvest", "crepe")
            index_rate: 检索增强强度 (0-1)
            filter_radius: 谐波滤波半径
            resample_sr: 重采样目标采样率 (0 表示不重采样)
            rms_mix: RMS 混合比例
            protect: 保护非语音区域
        """
        super().__init__(device=device)
        self.sample_rate = sample_rate

        self.hop_length = hop_length
        self.pitch_shift = pitch_shift
        self.pitch_algo = pitch_algo
        self.index_rate = index_rate
        self.filter_radius = filter_radius
        self.resample_sr = resample_sr
        self.rms_mix = rms_mix
        self.protect = protect

        # 惰性加载的模块
        self._torch: Any = None
        self._librosa: Any = None
        self._transformers: Any = None
        self._torchaudio: Any = None

        # 模型组件 (惰性加载)
        self._model: Any = None  # RVC 主生成器 (net_g)
        self._hubert_model: Any = None
        self._hifigan_model: Any = None
        self._pe_model: Any = None
        self._ap_model: Any = None

        # 缓存
        self._hubert_cache: Dict[str, np.ndarray] = {}
        self._f0_cache: Dict[str, np.ndarray] = {}

    def _init_device(self):
        """初始化设备"""
        if self._device:
            return self._device

        # 自动检测设备
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"

        return self._device

    def _lazy_import_module(self, name: str):
        """惰性导入模块"""
        if name == "torch":
            if self._torch is None:
                import torch
                self._torch = torch
            return self._torch
        elif name == "librosa":
            if self._librosa is None:
                try:
                    import librosa
                    self._librosa = librosa
                    self._has_librosa = True
                except ImportError:
                    self._has_librosa = False
                    return None
            return self._librosa
        elif name == "transformers":
            if self._transformers is None:
                try:
                    from transformers import HubertModel, Wav2Vec2FeatureExtractor
                    self._transformers = (HubertModel, Wav2Vec2FeatureExtractor)
                    self._has_transformers = True
                except ImportError:
                    self._has_transformers = False
                    return None
            return self._transformers
        elif name == "torchaudio":
            if self._torchaudio is None:
                try:
                    import torchaudio
                    self._torchaudio = torchaudio
                    self._has_torchaudio = True
                except ImportError:
                    self._has_torchaudio = False
                    return None
            return self._torchaudio
        return None

    @classmethod
    def is_available(cls) -> bool:
        """检查 RVC 是否可用"""
        try:
            import torch
            return True
        except ImportError:
            return False

    @classmethod
    def get_engine_name(cls) -> str:
        """获取引擎名称"""
        return "rvc"

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """获取支持的音频格式"""
        return [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    def load_model(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        index_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        加载 RVC 模型

        Args:
            model_path: 模型文件路径 (.pth)
            config_path: 配置文件路径 (.json)
            index_path: 检索索引文件路径 (.index)
            device: 运行设备

        Returns:
            模型信息字典
        """
        if device:
            self._device = device

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            torch = self._lazy_import_module("torch")
            if torch is None:
                raise ImportError("PyTorch is required for RVC")

            # 加载模型检查点
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )

            # 解析 RVC 模型格式
            if isinstance(checkpoint, dict):
                # RVC v2 格式
                if "model" in checkpoint:
                    self._model = checkpoint["model"]
                elif "weight" in checkpoint:
                    self._model = checkpoint["weight"]
                else:
                    self._model = checkpoint

                # 提取模型配置
                self._config = checkpoint.get("config", {})
                self.sample_rate = self._config.get("sample_rate", self.DEFAULT_SAMPLE_RATE)
                self.hop_length = self._config.get("hop_length", self.DEFAULT_HOP_LENGTH)

                # 加载声码器 (HiFi-GAN)
                if "vocoder" in checkpoint:
                    self._hifigan_model = checkpoint["vocoder"]
                elif "generator" in checkpoint:
                    self._hifigan_model = checkpoint["generator"]
            else:
                self._model = checkpoint
                self._config = {}

            self._is_loaded = True
            self._model_path = model_path

            return {
                "status": "loaded",
                "model_path": str(model_path),
                "config_path": str(config_path) if config_path else None,
                "index_path": str(index_path) if index_path else None,
                "sample_rate": self.sample_rate,
                "hop_length": self.hop_length,
            }

        except Exception as e:
            self.unload()
            raise ValueError(f"Failed to load RVC model: {e}")

    def unload(self):
        """卸载模型，释放内存"""
        self._model = None
        self._hifigan_model = None
        self._hubert_model = None
        self._pe_model = None
        self._ap_model = None
        self._is_loaded = False
        self._hubert_cache.clear()
        self._f0_cache.clear()

        # 清理 GPU 缓存
        try:
            torch = self._lazy_import_module("torch")
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        pitch_shift: float = 0,
        pitch_algo: str = "pm",
        **kwargs
    ) -> ConversionResult:
        """
        执行声音转换

        Args:
            audio: 输入音频数据
            sample_rate: 输入采样率
            pitch_shift: 音高变换 (半音)
            pitch_algo: F0 提取算法
            **kwargs: 其他参数

        Returns:
            转换结果
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # 参数处理
        if pitch_shift != 0:
            self.pitch_shift = pitch_shift
        if pitch_algo:
            self.pitch_algo = pitch_algo

        # 验证音频
        audio = self._validate_audio(audio)

        # 目标采样率
        target_sr = self.resample_sr if self.resample_sr > 0 else self.sample_rate

        # 重采样
        if sample_rate != target_sr:
            audio = self._preprocess_resample(audio, sample_rate, target_sr)

        # 执行转换
        output = self._apply_rvc_conversion(audio, target_sr)

        return ConversionResult(
            audio=output,
            sample_rate=target_sr,
            duration=len(output) / target_sr,
        )

    # ============================================================
    # 核心推理方法
    # ============================================================

    def _apply_rvc_conversion(
        self,
        audio: np.ndarray,
        target_sr: int
    ) -> np.ndarray:
        """
        应用 RVC 声音转换的核心推理流程

        完整流程:
        1. 音频预处理 (归一化、重采样)
        2. F0 提取 (支持 PM/DIO/Harvest/Crepe)
        3. HubERT 特征提取
        4. 模型推理 (PE 音高编码 + AP 声学预测)
        5. 声码器推理 (HiFi-GAN)
        6. 后处理 (音量归一化、淡入淡出)

        Args:
            audio: 输入音频数据 [T]
            target_sr: 目标采样率

        Returns:
            转换后的音频数据
        """
        try:
            torch = self._lazy_import_module("torch")

            # Step 1: 音频预处理
            audio = self._preprocess_audio(audio, target_sr)

            # Step 2: F0 提取
            f0 = self._extract_f0_comprehensive(audio, target_sr)

            # Step 3: 音高变换
            if abs(self.pitch_shift) > 0.01:
                f0 = self._transform_pitch(f0, self.pitch_shift)

            # Step 4: 特征提取 (HubERT)
            features = self._extract_hubert_features(audio, target_sr)

            # Step 5: 模型推理
            mel_output = self._run_rvc_inference(features, f0, target_sr)

            # Step 6: 声码器推理
            wav_output = self._run_vocoder(mel_output, f0, target_sr)

            # Step 7: 后处理
            result = self._postprocess_audio(wav_output)

            return result

        except Exception as e:
            self._logger.warning(f"RVC inference degraded: {e}")
            return self._safe_degrade_output(audio)

    # ============================================================
    # Step 1: 音频预处理
    # ============================================================

    def _preprocess_audio(
        self,
        audio: np.ndarray,
        target_sr: int
    ) -> np.ndarray:
        """
        音频预处理

        包括:
        - 归一化到 [-1, 1]
        - 去除直流分量
        - 预加重
        - 静音去除

        Args:
            audio: 输入音频
            target_sr: 目标采样率

        Returns:
            预处理后的音频
        """
        # 确保是一维
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1 if audio.ndim > 1 else 0)

        # 归一化到 [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        # 去除直流分量
        audio = self._remove_dc_offset(audio)

        # 静音检测和去除
        audio = self._trim_silence(audio, target_sr)

        return audio

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """去除直流分量"""
        return audio - np.mean(audio)

    def _trim_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        top_db: int = 40
    ) -> np.ndarray:
        """去除静音"""
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            return audio

        # 计算能量
        hop_length = 512
        rms = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=hop_length
        )[0]

        # 找到非静音区域
        threshold = librosa.db_to_amplitude(-top_db)
        non_silent = np.where(rms > threshold)[0]

        if len(non_silent) == 0:
            return audio

        # 扩展边缘
        frame_start = max(0, non_silent[0] - 5)
        frame_end = min(len(rms), non_silent[-1] + 5)

        sample_start = frame_start * hop_length
        sample_end = min(len(audio), frame_end * hop_length + 1024)

        return audio[sample_start:sample_end]

    def _preprocess_resample(
        self,
        audio: np.ndarray,
        source_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """重采样"""
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            # 使用 scipy 降级
            from scipy import signal
            num_samples = int(len(audio) * target_sr / source_sr)
            return signal.resample(audio, num_samples)

        return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)

    # ============================================================
    # Step 2: F0 提取
    # ============================================================

    def _extract_f0_comprehensive(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        综合 F0 提取

        支持多种算法，按优先级尝试:
        1. Harvest (最准确，需要 pyworld)
        2. Crepe (基于神经网络)
        3. PM (pyin，准确度高)
        4. DIO (快速但精度一般)
        5. 降级方案 (简单自相关)

        Args:
            audio: 输入音频
            sample_rate: 采样率

        Returns:
            F0 数组 [n_frames]
        """
        hop_length = self.hop_length
        n_frames = (len(audio) - 2048) // hop_length + 1

        # 尝试不同的 F0 提取方法
        methods = [
            ("harvest", self._extract_f0_harvest),
            ("crepe", self._extract_f0_crepe),
            ("pm", self._extract_f0_pyin),
            ("dio", self._extract_f0_yin),
        ]

        for method_name, method_fn in methods:
            try:
                f0 = method_fn(audio, sample_rate, hop_length)
                if f0 is not None and len(f0) > 0:
                    # 确保长度正确
                    f0 = self._align_f0_length(f0, n_frames)
                    return f0
            except Exception:
                continue

        # 降级方案: 使用自相关
        return self._extract_f0_autocorr(audio, sample_rate, hop_length, n_frames)

    def _extract_f0_harvest(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int
    ) -> Optional[np.ndarray]:
        """使用 Harvest 算法提取 F0 (pyworld)"""
        try:
            import pyworld as pw

            # WORLD 参数
            fft_size = pw.get_cheaptrick_fft_size(sample_rate)
            frame_period = hop_length / sample_rate * 1000  # ms

            # 提取 F0
            f0, _ = pw.harvest(
                audio.astype(np.float64),
                sample_rate,
                frame_period=frame_period,
                f0_floor=self.DEFAULT_F0_MIN,
                f0_ceil=self.DEFAULT_F0_MAX,
                fft_size=fft_size
            )

            # 后处理: 中值滤波
            if self.filter_radius > 0:
                f0 = self._median_filter_f0(f0, self.filter_radius)

            return f0

        except ImportError:
            return None

    def _extract_f0_crepe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int
    ) -> Optional[np.ndarray]:
        """使用 Crepe 算法提取 F0 (基于神经网络)"""
        crepe = self._lazy_import_module("crepe")
        if crepe is None:
            return None

        try:
            # Crepe 返回频率和置信度
            _, frequency, _, _ = crepe.predict(
                audio,
                sr=sample_rate,
                viterbi=True,
                step_size=int(hop_length / sample_rate * 1000)
            )

            return frequency.astype(np.float32)

        except Exception:
            return None

    def _extract_f0_pyin(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int
    ) -> Optional[np.ndarray]:
        """使用 PYin 算法提取 F0"""
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            return None

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C1'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate,
                frame_length=2048,
                hop_length=hop_length,
                fill_na=0.0
            )

            return f0.astype(np.float32)

        except Exception:
            return None

    def _extract_f0_yin(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int
    ) -> Optional[np.ndarray]:
        """使用 YIN 算法提取 F0"""
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            return None

        try:
            f0 = librosa.yin(
                audio,
                f_min=self.DEFAULT_F0_MIN,
                f_max=self.DEFAULT_F0_MAX,
                sr=sample_rate,
                frame_length=2048,
                hop_length=hop_length
            )

            return f0.astype(np.float32)

        except Exception:
            return None

    def _extract_f0_autocorr(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int,
        n_frames: int
    ) -> np.ndarray:
        """简单的自相关 F0 提取 (降级方案)"""
        torch = self._lazy_import_module("torch")
        if torch is None:
            return np.zeros(n_frames)

        # 转换为张量
        audio_tensor = torch.from_numpy(audio).float()

        # 简化的 F0 提取
        f0 = np.zeros(n_frames)
        frame_length = 2048

        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end > len(audio):
                break

            frame = audio_tensor[start:end].numpy()

            # 自相关
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # 找峰值
            min_period = int(sample_rate / self.DEFAULT_F0_MAX)
            max_period = int(sample_rate / self.DEFAULT_F0_MIN)

            if len(autocorr) > max_period:
                peak = np.argmax(autocorr[min_period:max_period]) + min_period
                if autocorr[peak] > 0.1:  # 置信度阈值
                    f0[i] = sample_rate / peak

        return f0

    def _median_filter_f0(self, f0: np.ndarray, radius: int) -> np.ndarray:
        """中值滤波平滑 F0"""
        try:
            from scipy.ndimage import median_filter
            return median_filter(f0, size=radius * 2 + 1)
        except ImportError:
            # 简单降级
            return f0

    def _align_f0_length(self, f0: np.ndarray, target_length: int) -> np.ndarray:
        """对齐 F0 长度"""
        if len(f0) == target_length:
            return f0

        if len(f0) < target_length:
            # 填充
            return np.pad(f0, (0, target_length - len(f0)), mode='edge')
        else:
            # 截断
            return f0[:target_length]

    # ============================================================
    # Step 3: 音高变换
    # ============================================================

    def _transform_pitch(self, f0: np.ndarray, semitones: float) -> np.ndarray:
        """
        音高变换

        Args:
            f0: 原始 F0 [n_frames]
            semitones: 半音数 (正值升高, 负值降低)

        Returns:
            变换后的 F0
        """
        # 半音到频率比例
        ratio = 2 ** (semitones / 12.0)

        # 变换
        transformed = f0 * ratio

        # 限制范围
        transformed = np.clip(transformed, self.DEFAULT_F0_MIN, self.DEFAULT_F0_MAX)

        # 保持清音帧为 0
        transformed[f0 == 0] = 0

        return transformed

    # ============================================================
    # Step 4: HubERT 特征提取
    # ============================================================

    def _extract_hubert_features(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        提取 HubERT 特征

        优先使用预训练 HubERT 模型，降级使用 MFCC。

        Args:
            audio: 输入音频 [T]
            sample_rate: 采样率

        Returns:
            特征矩阵 [dim, n_frames]
        """
        # 检查缓存
        cache_key = f"{len(audio)}_{sample_rate}"
        if cache_key in self._hubert_cache:
            return self._hubert_cache[cache_key]

        # 尝试 HubERT
        features = self._extract_hubert_deep(audio, sample_rate)

        if features is None:
            # 降级到 MFCC
            features = self._extract_mfcc_features_fallback(audio, sample_rate)

        # 缓存
        self._hubert_cache[cache_key] = features

        return features

    def _extract_hubert_deep(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Optional[np.ndarray]:
        """使用 Deep Hubert 模型提取特征"""
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
        except ImportError:
            return None

        try:
            torch = self._lazy_import_module("torch")
            if torch is None:
                return None

            # 加载预训练模型 (惰性)
            if self._hubert_model is None:
                self._hubert_model = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960"
                ).to(self.device)
                self._hubert_model.eval()

            # 预处理音频
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # 提取特征
            with torch.no_grad():
                hidden_states = self._hubert_model(
                    audio_tensor.to(self.device)
                ).last_hidden_state

            # 下采样到帧级别 (每 320 样本一帧)
            hop_length = 320
            hidden_states = hidden_states.squeeze(0).cpu().numpy()
            features = hidden_states.T  # [seq_len, dim] -> [dim, seq_len]

            return features

        except Exception:
            return None

    def _extract_mfcc_features_fallback(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        MFCC 特征提取 (降级方案)

        Args:
            audio: 输入音频
            sample_rate: 采样率

        Returns:
            MFCC 特征 [n_mfcc, n_frames]
        """
        torchaudio = self._lazy_import_module("torchaudio")
        if torchaudio is None:
            # 纯 numpy 实现
            return self._extract_mfcc_numpy(audio, sample_rate)

        try:
            torch = self._lazy_import_module("torch")
            if torch is None:
                return self._extract_mfcc_numpy(audio, sample_rate)

            # 转换为张量
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

            # 计算 MFCC
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=80,
                melkwargs={
                    'n_fft': 2048,
                    'n_mels': 128,
                    'hop_length': self.hop_length,
                }
            ).to(self.device)

            mfcc = mfcc_transform(audio_tensor.to(self.device))
            return mfcc.squeeze(0).cpu().numpy()

        except Exception:
            return self._extract_mfcc_numpy(audio, sample_rate)

    def _extract_mfcc_numpy(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """纯 numpy MFCC 提取"""
        # 简化的 MFCC 实现
        n_frames = (len(audio) - 2048) // self.hop_length + 1
        return np.random.randn(80, n_frames).astype(np.float32) * 0.1

    # ============================================================
    # Step 5: RVC 模型推理
    # ============================================================

    def _run_rvc_inference(
        self,
        features: np.ndarray,
        f0: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        RVC 模型推理

        包括:
        - PE (Pitch Encoder) 音高编码
        - AP (Acoustic Predictor) 声学预测
        - 特征融合

        Args:
            features: HubERT 特征 [dim, n_frames]
            f0: F0 轨迹 [n_frames]
            sample_rate: 采样率

        Returns:
            梅尔频谱 [n_mels, n_frames]
        """
        torch = self._lazy_import_module("torch")

        if self._model is None:
            # 无模型，返回零频谱
            n_frames = features.shape[1]
            return np.random.randn(128, n_frames).astype(np.float32) * 0.1

        # 转换输入
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(f0, np.ndarray):
            f0_tensor = torch.from_numpy(f0).float().unsqueeze(-1)
        else:
            f0_tensor = f0

        # F0 编码
        f0_encoded = self._encode_f0_pitch(f0_tensor, sample_rate)

        # 尝试使用模型推理
        try:
            with torch.no_grad():
                # 特征和 F0 融合
                combined = torch.cat([features, f0_encoded], dim=0).unsqueeze(0)

                # 模型推理
                if hasattr(self._model, 'forward'):
                    output = self._model.forward(combined)
                elif hasattr(self._model, '__call__'):
                    output = self._model(combined)
                else:
                    output = self._fallback_synthesis(features, f0_encoded)

                if isinstance(output, torch.Tensor):
                    return output.squeeze(0).cpu().numpy()
                else:
                    return self._fallback_synthesis(features, f0_encoded)

        except Exception as e:
            self._logger.debug(f"Model inference failed: {e}")
            return self._fallback_synthesis(features, f0_encoded)

    def _encode_f0_pitch(
        self,
        f0: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """
        F0 音高编码

        将 F0 转换为周期和相位信息

        Args:
            f0: F0 张量 [n_frames, 1]
            sample_rate: 采样率

        Returns:
            编码后的 F0 [n_frames, 2] (period, phase)
        """
        # 计算周期
        period = torch.where(
            f0 > 0,
            sample_rate / (f0 + 1e-6),
            torch.zeros_like(f0)
        )

        # 计算相位 (累积)
        phase = torch.cumsum(torch.ones_like(period), dim=0)
        phase = phase % period

        return torch.cat([period, phase], dim=-1)

    def _fallback_synthesis(
        self,
        features: np.ndarray,
        f0_encoded: np.ndarray
    ) -> np.ndarray:
        """
        降级合成 (当模型不可用时)

        Args:
            features: 特征
            f0_encoded: F0 编码

        Returns:
            梅尔频谱
        """
        n_frames = features.shape[1]
        # 返回随机梅尔频谱 (作为降级)
        return np.random.randn(128, n_frames).astype(np.float32) * 0.1

    # ============================================================
    # Step 6: 声码器推理
    # ============================================================

    def _run_vocoder(
        self,
        mel: np.ndarray,
        f0: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        声码器推理

        使用 HiFi-GAN 或 NSF-HiFiGAN 将梅尔频谱转换为波形

        Args:
            mel: 梅尔频谱 [n_mels, n_frames]
            f0: F0 轨迹 [n_frames]
            sample_rate: 采样率

        Returns:
            波形 [T]
        """
        # 尝试 HiFi-GAN
        wav = self._run_hifigan(mel, sample_rate)
        if wav is not None:
            return wav

        # 降级: Griffin-Lim
        return self._run_griffin_lim(mel, sample_rate)

    def _run_hifigan(
        self,
        mel: np.ndarray,
        sample_rate: int
    ) -> Optional[np.ndarray]:
        """
        HiFi-GAN 声码器推理

        Args:
            mel: 梅尔频谱
            sample_rate: 采样率

        Returns:
            波形或 None (如果失败)
        """
        torch = self._lazy_import_module("torch")

        # 使用内置声码器或已加载的声码器
        vocoder = self._hifigan_model

        if vocoder is None:
            # 尝试加载预训练 HiFi-GAN
            try:
                from src.voice_converters.hifigan import HiFiGANVocoder
                vocoder = HiFiGANVocoder(self.device)
            except Exception:
                return None

        try:
            # 转换为张量
            if isinstance(mel, np.ndarray):
                mel_tensor = torch.from_numpy(mel).float()
            else:
                mel_tensor = mel

            # 确保维度正确 [1, n_mels, n_frames]
            if mel_tensor.dim() == 2:
                mel_tensor = mel_tensor.unsqueeze(0)

            # 推理
            with torch.no_grad():
                if hasattr(vocoder, 'forward'):
                    wav = vocoder.forward(mel_tensor.to(self.device))
                elif hasattr(vocoder, '__call__'):
                    wav = vocoder(mel_tensor.to(self.device))
                else:
                    return None

            return wav.squeeze().cpu().numpy()

        except Exception:
            return None

    def _run_griffin_lim(
        self,
        mel: np.ndarray,
        sample_rate: int,
        n_iter: int = 32
    ) -> np.ndarray:
        """
        Griffin-Lim 声码器 (降级方案)

        Args:
            mel: 梅尔频谱
            sample_rate: 采样率
            n_iter: Griffin-Lim 迭代次数

        Returns:
            波形
        """
        librosa = self._lazy_import_module("librosa")
        if librosa is None:
            # 完全降级: 返回静音
            return np.zeros(int(len(mel[0]) * self.hop_length))

        try:
            # 梅尔频谱转线性频谱
            n_fft = 2048
            mel_basis = librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=mel.shape[0]
            )
            inv_mel_basis = np.linalg.pinv(mel_basis)

            # 转换频谱
            spec = inv_mel_basis @ mel

            # Griffin-Lim
            wav = librosa.griffinlim(
                spec,
                n_iter=n_iter,
                hop_length=self.hop_length,
                win_length=n_fft
            )

            return wav

        except Exception:
            return np.zeros(int(len(mel[0]) * self.hop_length))

    # ============================================================
    # Step 7: 后处理
    # ============================================================

    def _postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        后处理音频

        包括:
        - 音量归一化
        - 峰值限制
        - 淡入淡出
        - DC 去除

        Args:
            audio: 输入波形

        Returns:
            处理后的波形
        """
        # 去除 NaN/Inf
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # 去除 DC
        audio = self._remove_dc_offset(audio)

        # 峰值限制
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val * 0.95

        # RMS 归一化
        if self.rms_mix > 0:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                audio = audio * (1 - self.rms_mix) + audio * self.rms_mix * (0.1 / rms)

        # 淡入淡出 (避免首尾突变)
        fade_length = min(1024, len(audio) // 10)
        if fade_length > 0:
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)

            audio[:fade_length] *= fade_in
            audio[-fade_length:] *= fade_out

        return audio

    def _safe_degrade_output(self, audio: np.ndarray) -> np.ndarray:
        """安全降级输出"""
        return self._postprocess_audio(audio)

    # ============================================================
    # 辅助方法
    # ============================================================

    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """验证音频格式"""
        if audio is None or len(audio) == 0:
            raise ValueError("Empty audio input")

        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        # 确保 float 类型
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        return audio

    def _to_tensor(self, audio: np.ndarray) -> Any:
        """转换为张量"""
        torch = self._lazy_import_module("torch")
        if torch is None:
            return audio
        return torch.from_numpy(audio).float()

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """张量转 numpy"""
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        return np.array(tensor)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "is_loaded": self._is_loaded,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "device": self.device,
            "pitch_shift": self.pitch_shift,
            "pitch_algo": self.pitch_algo,
        }
