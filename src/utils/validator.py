"""
Validator - 参数校验工具
提供音频处理相关参数的验证
"""

from typing import Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from src.exceptions import SOMAError


class ValidationError(SOMAError):
    """验证错误"""
    pass


class AudioValidator:
    """
    音频参数验证器
    
    验证音频处理相关的输入参数
    """
    
    # 支持的采样率
    VALID_SAMPLE_RATES = {
        8000, 11025, 16000, 22050, 32000, 
        44100, 48000, 88200, 96000, 176400, 192000
    }
    
    # 支持的声道数
    VALID_CHANNELS = {1, 2, 6, 8}  # mono, stereo, 5.1, 7.1
    
    # 支持的音频格式
    VALID_FORMATS = {
        "wav", "mp3", "flac", "ogg", "aac", 
        "m4a", "wma", "aiff", "amr", "opus"
    }
    
    @classmethod
    def validate_audio_path(cls, path: str, must_exist: bool = True) -> Path:
        """
        验证音频文件路径
        
        Args:
            path: 文件路径
            must_exist: 是否必须存在
            
        Returns:
            Path 对象
            
        Raises:
            ValidationError: 路径无效
        """
        if not path:
            raise ValidationError("File path cannot be empty")
        
        file_path = Path(path)
        
        if must_exist and not file_path.exists():
            raise ValidationError(f"File does not exist: {path}")
        
        if must_exist and not file_path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        
        # 检查扩展名
        suffix = file_path.suffix[1:].lower()
        if suffix not in cls.VALID_FORMATS:
            raise ValidationError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(sorted(cls.VALID_FORMATS))}"
            )
        
        return file_path
    
    @classmethod
    def validate_sample_rate(cls, sample_rate: int) -> int:
        """
        验证采样率
        
        Args:
            sample_rate: 采样率
            
        Returns:
            验证后的采样率
            
        Raises:
            ValidationError: 采样率无效
        """
        if not isinstance(sample_rate, int):
            raise ValidationError(f"Sample rate must be integer, got {type(sample_rate)}")
        
        if sample_rate <= 0:
            raise ValidationError(f"Sample rate must be positive, got {sample_rate}")
        
        if sample_rate not in cls.VALID_SAMPLE_RATES:
            # 警告但不抛出错误
            print(f"Warning: Non-standard sample rate {sample_rate}Hz")
        
        return sample_rate
    
    @classmethod
    def validate_channels(cls, channels: int) -> int:
        """
        验证声道数
        
        Args:
            channels: 声道数
            
        Returns:
            验证后的声道数
            
        Raises:
            ValidationError: 声道数无效
        """
        if not isinstance(channels, int):
            raise ValidationError(f"Channels must be integer, got {type(channels)}")
        
        if channels <= 0:
            raise ValidationError(f"Channels must be positive, got {channels}")
        
        if channels not in cls.VALID_CHANNELS:
            raise ValidationError(
                f"Unsupported channels: {channels}. "
                f"Supported: {', '.join(map(str, sorted(cls.VALID_CHANNELS)))}"
            )
        
        return channels
    
    @classmethod
    def validate_audio_array(cls, audio: np.ndarray) -> np.ndarray:
        """
        验证音频数组
        
        Args:
            audio: 音频数据
            
        Returns:
            验证后的音频数组
            
        Raises:
            ValidationError: 音频数据无效
        """
        if audio is None:
            raise ValidationError("Audio data cannot be None")
        
        if not isinstance(audio, np.ndarray):
            try:
                audio = np.array(audio)
            except Exception:
                raise ValidationError("Audio data cannot be converted to numpy array")
        
        if audio.ndim == 0:
            raise ValidationError("Audio data is scalar, expected array")
        
        if audio.ndim > 2:
            raise ValidationError(f"Audio data has too many dimensions: {audio.ndim}")
        
        if len(audio) == 0:
            raise ValidationError("Audio data is empty")
        
        return audio
    
    @classmethod
    def validate_frequency(cls, freq: float, min_freq: float = 0, max_freq: float = 20000) -> float:
        """
        验证频率参数
        
        Args:
            freq: 频率值
            min_freq: 最小频率
            max_freq: 最大频率
            
        Returns:
            验证后的频率
            
        Raises:
            ValidationError: 频率无效
        """
        if not isinstance(freq, (int, float)):
            raise ValidationError(f"Frequency must be number, got {type(freq)}")
        
        if freq < min_freq or freq > max_freq:
            raise ValidationError(
                f"Frequency {freq}Hz out of range [{min_freq}, {max_freq}]Hz"
            )
        
        return float(freq)
    
    @classmethod
    def validate_gain(cls, gain: float, min_gain: float = -60, max_gain: float = 60) -> float:
        """
        验证增益参数
        
        Args:
            gain: 增益值(dB)
            min_gain: 最小增益
            max_gain: 最大增益
            
        Returns:
            验证后的增益
            
        Raises:
            ValidationError: 增益无效
        """
        if not isinstance(gain, (int, float)):
            raise ValidationError(f"Gain must be number, got {type(gain)}")
        
        if gain < min_gain or gain > max_gain:
            raise ValidationError(
                f"Gain {gain}dB out of range [{min_gain}, {max_gain}]dB"
            )
        
        return float(gain)


class EffectParameterValidator:
    """
    效果器参数验证器
    """
    
    @classmethod
    def validate_eq_params(cls, params: dict) -> dict:
        """
        验证均衡器参数
        
        Args:
            params: 参数字典
            
        Returns:
            验证后的参数
            
        Raises:
            ValidationError: 参数无效
        """
        if "bands" in params:
            bands = params["bands"]
            if not isinstance(bands, list):
                raise ValidationError("EQ bands must be a list")
            
            for i, band in enumerate(bands):
                if not isinstance(band, dict):
                    raise ValidationError(f"Band {i} must be a dictionary")
                
                if "freq" in band:
                    AudioValidator.validate_frequency(band["freq"], 20, 20000)
                
                if "gain" in band:
                    AudioValidator.validate_gain(band["gain"], -20, 20)
                
                if "q" in band:
                    q = band["q"]
                    if not 0.1 <= q <= 10:
                        raise ValidationError(f"Band Q must be in [0.1, 10], got {q}")
        
        return params
    
    @classmethod
    def validate_reverb_params(cls, params: dict) -> dict:
        """
        验证混响参数
        
        Args:
            params: 参数字典
            
        Returns:
            验证后的参数
        """
        # 验证 room_size
        if "room_size" in params:
            room_size = params["room_size"]
            if not 0.0 <= room_size <= 1.0:
                raise ValidationError(f"room_size must be in [0, 1], got {room_size}")
        
        # 验证 damping
        if "damping" in params:
            damping = params["damping"]
            if not 0.0 <= damping <= 1.0:
                raise ValidationError(f"damping must be in [0, 1], got {damping}")
        
        # 验证 wet_level 和 dry_level
        for key in ["wet_level", "dry_level"]:
            if key in params:
                level = params[key]
                if not 0.0 <= level <= 1.0:
                    raise ValidationError(f"{key} must be in [0, 1], got {level}")
        
        return params
    
    @classmethod
    def validate_pitch_params(cls, params: dict) -> dict:
        """
        验证音调参数
        
        Args:
            params: 参数字典
            
        Returns:
            验证后的参数
        """
        # 验证 semitones
        if "semitones" in params:
            semitones = params["semitones"]
            if not -24 <= semitones <= 24:
                raise ValidationError(f"semitones must be in [-24, 24], got {semitones}")
        
        # 验证 cents
        if "cents" in params:
            cents = params["cents"]
            if not -100 <= cents <= 100:
                raise ValidationError(f"cents must be in [-100, 100], got {cents}")
        
        return params


class PipelineValidator:
    """
    流水线参数验证器
    """
    
    @classmethod
    def validate_node_config(cls, config: dict) -> dict:
        """
        验证节点配置
        
        Args:
            config: 节点配置
            
        Returns:
            验证后的配置
            
        Raises:
            ValidationError: 配置无效
        """
        required_fields = ["name", "type", "params"]
        
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field: {field}")
        
        # 验证节点类型
        valid_types = {"separator", "effect", "converter", "filter", "custom"}
        if config["type"] not in valid_types:
            raise ValidationError(
                f"Invalid node type: {config['type']}. "
                f"Valid types: {', '.join(valid_types)}"
            )
        
        return config
    
    @classmethod
    def validate_pipeline_config(cls, config: dict) -> dict:
        """
        验证流水线配置
        
        Args:
            config: 流水线配置
            
        Returns:
            验证后的配置
        """
        if "nodes" not in config:
            raise ValidationError("Pipeline config must contain 'nodes' field")
        
        if not isinstance(config["nodes"], list):
            raise ValidationError("Pipeline nodes must be a list")
        
        for i, node_config in enumerate(config["nodes"]):
            try:
                cls.validate_node_config(node_config)
            except ValidationError as e:
                raise ValidationError(f"Node {i}: {str(e)}")
        
        return config


def validate(value: Any, rules: dict) -> Tuple[bool, Optional[str]]:
    """
    通用验证函数
    
    Args:
        value: 待验证的值
        rules: 验证规则
        
    Returns:
        (is_valid, error_message)
    """
    try:
        if "type" in rules:
            expected_type = rules["type"]
            if not isinstance(value, expected_type):
                return False, f"Expected {expected_type.__name__}, got {type(value).__name__}"
        
        if "min" in rules and value < rules["min"]:
            return False, f"Value {value} is less than minimum {rules['min']}"
        
        if "max" in rules and value > rules["max"]:
            return False, f"Value {value} is greater than maximum {rules['max']}"
        
        if "choices" in rules and value not in rules["choices"]:
            return False, f"Value {value} not in allowed choices: {rules['choices']}"
        
        if "regex" in rules:
            import re
            if not re.match(rules["regex"], str(value)):
                return False, f"Value does not match pattern: {rules['regex']}"
        
        return True, None
        
    except Exception as e:
        return False, str(e)
