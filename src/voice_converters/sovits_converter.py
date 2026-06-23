"""
SoVITS Converter - So-VITS-SVC 4.1 声音转换引擎实现

基于 So-VITS-SVC 4.1 的声音转换实现。
支持扩散模式和非扩散模式。
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import numpy as np
import json

from .base import (
    BaseVoiceConverter,
    ConversionParams,
    ConversionResult,
    ModelInfo,
    ConverterType,
    F0Method,
    LazyImportMixin,
    EngineCapability,
)


class SoVITSDependencyError(ImportError):
    """SoVITS 依赖缺失错误"""
    pass


class SoVITSConverter(BaseVoiceConverter, LazyImportMixin, EngineCapability):
    """
    So-VITS-SVC 4.1 声音转换器
    
    基于 So-VITS-SVC (Singular Value Decomposition - Text-to-Speech Voice 
    Conversion System) 4.1 的实现。
    
    特点:
    - 支持扩散模式 (Diffusion)
    - 高保真音色转换
    - 支持 G_*.pth + config.json 模型格式
    - 说话人嵌入支持
    
    依赖:
    - torch
    - torchaudio
    - numpy
    
    可选依赖:
    - librosa (用于音频处理)
    - scipy (用于信号处理)
    - omegaconf (用于配置管理)
    """
    
    # SoVITS 支持的特性
    SUPPORTS_F0 = True
    SUPPORTS_TIMBRE_PROTECTION = True
    SUPPORTS_DIFFUSION = True
    SUPPORTS_SPEAKER_EMBEDDING = True
    MAX_SAMPLE_RATE = 48000
    RECOMMENDED_SAMPLE_RATE = 40000
    
    # SoVITS 支持的 f0 方法
    SUPPORTED_F0_METHODS = [
        F0Method.PM,
        F0Method.DIO,
        F0Method.CREPE,
        F0Method.HARVEST,
    ]
    
    # SoVITS 不需要索引文件
    REQUIRE_INDEX = False
    
    # 依赖包
    REQUIRED_PACKAGES = ["torch"]
    
    # SoVITS 默认参数
    DEFAULT_SAMPLE_RATE = 40000
    DEFAULT_HOP_LENGTH = 512  # SoVITS 帧移较大
    
    def __init__(
        self,
        device: Optional[str] = None,
        enable_diffusion: bool = False,
        diffusion_steps: int = 10,
    ):
        """
        初始化 SoVITS 转换器
        
        Args:
            device: 运行设备 ('cpu', 'cuda', 'mps')
            enable_diffusion: 是否启用扩散模式
            diffusion_steps: 扩散步数
        """
        super().__init__(device)
        
        # 扩散设置
        self.enable_diffusion = enable_diffusion
        self.diffusion_steps = diffusion_steps
        
        # SoVITS 组件
        self._hps = None              # 超参数
        self._net_g = None           # 生成器网络
        self._扩散_model = None      # 扩散模型 (可选)
        self._speaker_map = {}       # 说话人映射
        
        # 音频处理组件
        self._mel_transform = None   # 梅尔频谱变换器
        self._vocoder = None          # 声码器
    
    def load_model(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        diffusion_model_path: Optional[str] = None,
        diffusion_config_path: Optional[str] = None,
        speaker_id: int = 0,
        **kwargs
    ) -> ModelInfo:
        """
        加载 SoVITS 模型
        
        Args:
            model_path: 生成器模型路径 (G_*.pth)
            config_path: 配置文件路径 (config.json)
            diffusion_model_path: 扩散模型路径 (可选)
            diffusion_config_path: 扩散配置路径 (可选)
            speaker_id: 说话人 ID
            **kwargs: 其他参数
            
        Returns:
            ModelInfo: 模型信息
            
        Raises:
            FileNotFoundError: 模型文件不存在
            ValueError: 模型格式错误
        """
        # 检查文件存在
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"SoVITS model not found: {model_path}")
        
        # 如果没有提供 config_path，尝试查找
        if config_path is None:
            config_path = self._find_config_file(model_file)
        
        if config_path is None:
            raise ValueError("config.json not found. Please provide config_path.")
        
        # 检查依赖
        missing = self._check_all_dependencies()
        if missing:
            raise SoVITSDependencyError(
                f"SoVITS requires additional packages: {', '.join(missing)}\n"
                f"Install with: uv add {' '.join(missing)}"
            )
        
        # 延迟导入
        torch = self._lazy_import_module("torch")
        
        try:
            # 加载配置
            self._load_config(config_path)
            
            # 加载生成器模型
            self._load_generator(model_path, **kwargs)
            
            # 加载扩散模型 (如果启用)
            if self.enable_diffusion and diffusion_model_path:
                self._load_diffusion_model(diffusion_model_path, diffusion_config_path)
            
            # 初始化音频处理器
            self._init_audio_processor()
            
            # 设置设备
            if self.device != "cpu":
                self._net_g.to(self.device)
            
            self._is_loaded = True
            
            # 构建模型信息
            self._model_info = ModelInfo(
                name=model_file.stem,
                type=ConverterType.SOVITS,
                version=self._hps.get("version", "4.1") if self._hps else "4.1",
                sample_rate=self._hps.get("audio", {}).get("sample_rate", self.DEFAULT_SAMPLE_RATE) if self._hps else self.DEFAULT_SAMPLE_RATE,
                description=f"SoVITS 4.1 {'(Diffusion)' if self.enable_diffusion else ''}",
                file_path=str(model_file),
                config_path=config_path,
                is_loaded=True,
            )
            
            return self._model_info
            
        except Exception as e:
            self.unload()
            raise ValueError(f"Failed to load SoVITS model: {e}")
    
    def _find_config_file(self, model_file: Path) -> Optional[str]:
        """查找配置文件"""
        # 常见配置路径
        possible_paths = [
            model_file.parent / "config.json",
            model_file.parent / "configs" / "config.json",
            model_file.parent.parent / "config.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _load_config(self, config_path: str):
        """加载 SoVITS 配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self._hps = config
        
        # 解析配置
        if "train" in config:
            # 新版配置格式
            self.config = config.get("train", {})
            self.model_config = config.get("model", {})
            self.data_config = config.get("data", {})
        else:
            # 旧版配置格式
            self.config = config
        
        # 提取采样率
        if "audio" in config:
            self.sample_rate = config["audio"].get("sample_rate", self.DEFAULT_SAMPLE_RATE)
        elif "sampling_rate" in config:
            self.sample_rate = config["sampling_rate"]
        else:
            self.sample_rate = self.DEFAULT_SAMPLE_RATE
        
        # 构建说话人映射
        if "spk" in config:
            self._speaker_map = config["spk"]
        elif "n_speakers" in config:
            self._speaker_map = {i: f"Speaker_{i}" for i in range(config["n_speakers"])}
    
    def _load_generator(self, model_path: str, **kwargs):
        """加载生成器模型"""
        torch = self._lazy_import_module("torch")
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location="cpu")
        
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # TODO: 实例化生成器网络
        # 这里需要根据 config 创建网络结构
        # from .sovits_models import Generator
        # self._net_g = Generator(self.config)
        # self._net_g.load_state_dict(state_dict)
        
        # 简化: 存储权重
        self._state_dict = state_dict
    
    def _load_diffusion_model(
        self,
        model_path: str,
        config_path: Optional[str] = None
    ):
        """加载扩散模型"""
        if not Path(model_path).exists():
            return
        
        torch = self._lazy_import_module("torch")
        
        try:
            # 加载扩散模型
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # TODO: 实例化扩散模型
            # from .diffusion import DiffusionModel
            # self._diffusion_model = DiffusionModel(...)
            # self._diffusion_model.load_state_dict(checkpoint)
            
        except Exception:
            # 扩散模型加载失败，禁用扩散
            self.enable_diffusion = False
    
    def _init_audio_processor(self):
        """初始化音频处理器"""
        # 检查 librosa 是否可用
        try:
            import librosa
            self._has_librosa = True
        except ImportError:
            self._has_librosa = False
    
    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        params: Optional[ConversionParams] = None,
        speaker_id: int = 0,
        **kwargs
    ) -> ConversionResult:
        """
        执行 SoVITS 声音转换
        
        Args:
            audio: 输入音频
            sample_rate: 输入采样率
            params: 转换参数
            speaker_id: 说话人 ID
            **kwargs: 参数覆盖
            
        Returns:
            ConversionResult: 转换结果
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 验证输入
        audio = self._validate_audio(audio)
        params = self._validate_params(params)
        
        # 合并 kwargs 到 params
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(params, key):
                    setattr(params, key, value)
        
        try:
            # 重采样到模型采样率
            if sample_rate != params.sample_rate:
                audio = self._resample(audio, sample_rate, params.sample_rate)
            
            # 提取梅尔频谱
            mel_spec = self._extract_mel_spectrogram(audio, params.sample_rate)
            
            # 提取 F0
            f0 = self._extract_f0(audio, params.sample_rate, params.pitch_algo)
            
            # 计算音高变换
            pitch_factor = 2 ** (params.pitch_shift / 12)
            
            # 应用 SoVITS 转换
            if self.enable_diffusion:
                output_audio = self._apply_diffusion_conversion(
                    mel_spec, f0, pitch_factor, speaker_id, params
                )
            else:
                output_audio = self._apply_conversion(
                    mel_spec, f0, pitch_factor, speaker_id, params
                )
            
            # 创建结果
            result = self._create_result(
                output_audio,
                params.sample_rate,
                info={
                    "engine": "SoVITS",
                    "version": "4.1",
                    "diffusion": self.enable_diffusion,
                    "pitch_algo": params.pitch_algo,
                    "speaker_id": speaker_id,
                }
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"SoVITS conversion failed: {e}")
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """重采样"""
        if self._has_librosa:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        else:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples)
    
    def _extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        提取梅尔频谱
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            梅尔频谱
        """
        if self._has_librosa:
            import librosa
            
            # 获取参数
            n_fft = self._hps.get("audio", {}).get("filter_length", 2048) if self._hps else 2048
            hop_length = self._hps.get("audio", {}).get("hop_length", 512) if self._hps else 512
            win_length = self._hps.get("audio", {}).get("win_length", 2048) if self._hps else 2048
            n_mels = self._hps.get("audio", {}).get("mel_channels", 128) if self._hps else 128
            
            # 计算梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
            )
            
            # 转换为分贝
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
        else:
            # 返回零数组
            return np.zeros((128, len(audio) // 512))
    
    def _extract_f0(
        self,
        audio: np.ndarray,
        sample_rate: int,
        method: str
    ) -> np.ndarray:
        """
        提取基频 (F0)
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            method: 提取方法
            
        Returns:
            f0 数组
        """
        if self._has_librosa:
            import librosa
            
            hop_length = self._hps.get("audio", {}).get("hop_length", 512) if self._hps else 512
            
            if method == "pm":
                f0, voiced_flag = librosa.pyin(
                    audio,
                    fmin=librosa.note_to_hz('C1'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sample_rate,
                    frame_length=2048,
                    hop_length=hop_length,
                )
            elif method == "dio":
                f0, voiced_flag = librosa.yin(
                    audio,
                    fmin=librosa.note_to_hz('C1'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sample_rate,
                    hop_length=hop_length,
                )
            else:
                f0, voiced_flag = librosa.pyin(
                    audio,
                    fmin=librosa.note_to_hz('C1'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sample_rate,
                    hop_length=hop_length,
                )
            
            f0 = np.nan_to_num(f0, nan=0.0)
            
        else:
            hop_length = self._hps.get("audio", {}).get("hop_length", 512) if self._hps else 512
            n_frames = len(audio) // hop_length
            f0 = np.zeros(n_frames)
        
        return f0
    
    def _apply_conversion(
        self,
        mel_spec: np.ndarray,
        f0: np.ndarray,
        pitch_factor: float,
        speaker_id: int,
        params: ConversionParams
    ) -> np.ndarray:
        """
        应用 SoVITS 转换 (非扩散模式)
        
        这里实现实际的 SoVITS 推理逻辑。
        """
        torch = self._lazy_import_module("torch")
        
        # TODO: 完整的 SoVITS 推理实现
        # 1. F0 变换
        # 2. 说话人嵌入
        # 3. 生成器推理
        # 4. 声码器合成
        
        # 简化实现：返回零数组
        # 实际需要调用 self._net_g 进行推理
        output = np.zeros(len(f0) * 512)
        
        return output
    
    def _apply_diffusion_conversion(
        self,
        mel_spec: np.ndarray,
        f0: np.ndarray,
        pitch_factor: float,
        speaker_id: int,
        params: ConversionParams
    ) -> np.ndarray:
        """
        应用 SoVITS 转换 (扩散模式)
        
        使用扩散模型进行更高质量的转换
        """
        # 扩散模式需要在生成器前添加扩散去噪过程
        # TODO: 实现扩散推理
        
        # 先进行基础转换
        output = self._apply_conversion(mel_spec, f0, pitch_factor, speaker_id, params)
        
        # 然后应用扩散去噪
        # if self._diffusion_model is not None:
        #     output = self._diffusion_model.denoise(output, steps=params.diffusion_steps)
        
        return output
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """获取当前模型信息"""
        return self._model_info
    
    def unload(self):
        """卸载模型，释放显存"""
        # 清理生成器
        if self._net_g is not None:
            del self._net_g
            self._net_g = None
        
        # 清理扩散模型
        if self._扩散_model is not None:
            del self._扩散_model
            self._扩散_model = None
        
        # 清理声码器
        if self._vocoder is not None:
            del self._vocoder
            self._vocoder = None
        
        # 清理配置
        self._hps = None
        self._state_dict = None
        
        # 强制 GC
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self._is_loaded = False
    
    def set_speaker_id(self, speaker_id: int):
        """设置说话人 ID"""
        if speaker_id in self._speaker_map:
            self._current_speaker_id = speaker_id
        else:
            raise ValueError(f"Invalid speaker ID: {speaker_id}")
    
    def get_speaker_count(self) -> int:
        """获取模型支持的说话人数量"""
        return len(self._speaker_map) if self._speaker_map else 1
    
    def get_speaker_list(self) -> List[Dict[str, Any]]:
        """获取说话人列表"""
        return [
            {"id": k, "name": v}
            for k, v in self._speaker_map.items()
        ]
    
    def get_available_f0_methods(self) -> List[F0Method]:
        """获取可用的 f0 提取方法"""
        return self.SUPPORTED_F0_METHODS
    
    def set_diffusion(self, enable: bool, steps: int = 10):
        """
        设置扩散模式
        
        Args:
            enable: 是否启用
            steps: 扩散步数
        """
        if enable and self._扩散_model is None:
            print("Warning: Diffusion model not loaded. Diffusion disabled.")
            return
        
        self.enable_diffusion = enable
        self.diffusion_steps = steps
    
    def get_conversion_preset(self, preset_name: str) -> ConversionParams:
        """
        获取转换预设
        
        Args:
            preset_name: 预设名称
            
        Returns:
            ConversionParams: 预设参数
        """
        presets = {
            "quality": ConversionParams(
                pitch_shift=0,
                pitch_algo="dio",
                vpm=0.5,
                timbre_protection=0.5,
                rms_mix=0.5,
                diffusion_steps=20,
            ),
            "speed": ConversionParams(
                pitch_shift=0,
                pitch_algo="pm",
                vpm=0.5,
                timbre_protection=0.3,
                rms_mix=0.5,
                diffusion_steps=0,
            ),
            "natural": ConversionParams(
                pitch_shift=0,
                pitch_algo="harvest",
                vpm=0.3,
                timbre_protection=0.7,
                rms_mix=0.5,
                diffusion_steps=15,
            ),
            "diffusion": ConversionParams(
                pitch_shift=0,
                pitch_algo="dio",
                vpm=0.5,
                timbre_protection=0.5,
                rms_mix=0.5,
                diffusion_steps=20,
            ),
        }
        
        return presets.get(preset_name, ConversionParams())
    
    @classmethod
    def get_engine_name(cls) -> str:
        """获取引擎名称"""
        return "So-VITS-SVC 4.1"
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """获取支持的模型格式"""
        return [".pth"]
    
    @classmethod
    def is_available(cls) -> bool:
        """检查引擎是否可用"""
        try:
            import torch
            return True
        except ImportError:
            return False
