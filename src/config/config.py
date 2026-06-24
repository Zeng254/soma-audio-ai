"""
SOMA 配置中心 - 主配置类

提供统一的配置管理系统，支持：
- 从 JSON/YAML 文件加载和保存
- 层级覆盖：默认值 < 配置文件 < 用户输入
- 类型安全的 get/set 方法
- 配置验证和自动补全
"""

import os
import json
import copy
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar, Type
from dataclasses import is_dataclass, asdict

from .defaults import DEFAULT_CONFIG
from src.exceptions import ConfigError, ConfigLoadError, ConfigValidationError

T = TypeVar('T')

logger = logging.getLogger(__name__)


def _safe_asdict(obj):
    """安全地将 dataclass 转换为字典（处理嵌套）"""
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _safe_asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_asdict(item) for item in obj]
    if is_dataclass(obj):
        return {k: _safe_asdict(v) for k, v in asdict(obj).items()}
    return obj


class Config:
    """
    SOMA 配置管理类

    支持层级覆盖机制：
    1. 默认值 (defaults.py)
    2. 配置文件 (JSON/YAML)
    3. 用户输入 (运行时覆盖)

    示例:
        # 加载配置
        config = Config.load("~/.soma/config.json")

        # 获取值
        sample_rate = config.get("audio_utils.default_sample_rate")
        device = config.get("separators.device", default="cuda")

        # 设置值
        config.set("separators.device", "cuda")

        # 保存配置
        config.save()

        # 重置为默认
        config.reset()
    """

    def __init__(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        user_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化配置

        Args:
            base_config: 基础配置（默认配置）
            user_config: 用户配置（会覆盖基础配置）
        """
        if base_config is None:
            self._base = copy.deepcopy(DEFAULT_CONFIG)
        elif isinstance(base_config, dict):
            self._base = base_config
        elif is_dataclass(base_config) and not isinstance(base_config, type):
            # 如果是 dataclass 实例，转换为字典
            self._base = asdict(base_config)
        else:
            self._base = copy.deepcopy(DEFAULT_CONFIG)
        self._user = user_config or {}
        self._path: Optional[Path] = None
        self._loaded = False
        
        # 应用用户配置到基础配置
        if self._user:
            self._apply_user_config(self._base, self._user)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        auto_create: bool = True
    ) -> "Config":
        """
        从文件加载配置

        Args:
            path: 配置文件路径
            auto_create: 文件不存在时是否自动创建默认配置

        Returns:
            Config 实例
        """
        path = Path(path).expanduser()
        config = cls()

        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if path.suffix in ['.yaml', '.yml']:
                        import yaml
                        config._user = yaml.safe_load(f) or {}
                    else:
                        config._user = json.load(f)
                config._path = path
                config._loaded = True
                logger.info(f"配置已从 {path} 加载")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}，使用默认配置")
                config._user = {}
        elif auto_create:
            # 自动创建默认配置文件
            config._path = path
            config.save()
            logger.info(f"已创建默认配置文件: {path}")
        else:
            logger.info("使用默认配置")

        return config

    @classmethod
    def _apply_user_config(
        cls,
        base: Any,
        user: Dict[str, Any]
    ) -> None:
        """将用户配置应用到基础配置对象（支持字典和dataclass）"""
        # 处理 dataclass 对象
        if is_dataclass(base) and not isinstance(base, type):
            base_dict = asdict(base) if hasattr(base, '__dataclass_fields__') else {}
            for key, value in user.items():
                if hasattr(base, key):
                    current = getattr(base, key)
                    if isinstance(current, dict) and isinstance(value, dict):
                        cls._apply_dict(current, value)
                    elif value is not None:
                        # 类型安全转换
                        try:
                            if isinstance(current, bool):
                                if isinstance(value, str):
                                    value = value.lower() in ('true', '1', 'yes')
                                else:
                                    value = bool(value)
                            elif isinstance(current, int) and isinstance(value, (int, float)):
                                value = int(value)
                            elif isinstance(current, float) and isinstance(value, (int, float)):
                                value = float(value)
                            setattr(base, key, value)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"配置类型转换失败 {key}: {e}")
        # 处理字典对象
        elif isinstance(base, dict):
            for key, value in user.items():
                if key in base:
                    if isinstance(base[key], dict) and isinstance(value, dict):
                        cls._apply_dict(base[key], value)
                    elif value is not None:
                        base[key] = value

    @classmethod
    def _apply_dict(
        cls,
        base_dict: Dict[str, Any],
        user_dict: Dict[str, Any]
    ) -> None:
        """将字典值应用到字典对象"""
        for key, value in user_dict.items():
            if key in base_dict:
                if isinstance(base_dict[key], dict) and isinstance(value, dict):
                    cls._apply_dict(base_dict[key], value)
                elif value is not None:
                    # 安全类型转换
                    try:
                        current = base_dict[key]
                        if isinstance(current, bool):
                            if isinstance(value, str):
                                value = value.lower() in ('true', '1', 'yes')
                            else:
                                value = bool(value)
                        elif isinstance(current, int) and isinstance(value, (int, float)):
                            value = int(value)
                        elif isinstance(current, float) and isinstance(value, (int, float)):
                            value = float(value)
                        elif isinstance(current, list) and not isinstance(value, list):
                            value = [value]
                        base_dict[key] = value
                    except (ValueError, TypeError) as e:
                        logger.warning(f"配置类型转换失败 {key}: {e}")

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        保存配置到文件

        Args:
            path: 保存路径，默认使用加载时的路径
        """
        save_path = Path(path or self._path).expanduser() if self._path or path else None

        if not save_path:
            raise ConfigError("未指定保存路径")

        # 确保目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换配置为字典
        config_dict = self.to_dict()

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.suffix in ['.yaml', '.yml']:
                    import yaml
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到 {save_path}")
        except Exception as e:
            raise ConfigError(f"保存配置失败: {e}")

    def get(
        self,
        key: str,
        default: Optional[T] = None,
        value_type: Optional[Type[T]] = None
    ) -> Optional[T]:
        """
        获取配置值

        Args:
            key: 配置键，支持点号分隔的路径，如 "separators.device"
            default: 默认值
            value_type: 期望的类型

        Returns:
            配置值

        示例:
            config.get("audio_utils.default_sample_rate")
            config.get("separators.device", default="cuda")
        """
        # 先从用户配置获取
        value = self._get_nested(self._user, key)

        # 如果用户配置没有，从基础配置获取
        if value is None:
            value = self._get_nested(self._base, key)

        # 如果还是没有，返回默认值
        if value is None:
            return default

        # 类型转换
        if value_type is not None:
            if isinstance(value, value_type):
                return value
            try:
                if value_type == bool:
                    # 特殊处理 bool
                    if isinstance(value, str):
                        return value.lower() in ('true', '1', 'yes')
                    return bool(value)
                return value_type(value)
            except (ValueError, TypeError, AttributeError):
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        设置配置值

        Args:
            key: 配置键，支持点号分隔的路径
            value: 配置值
        """
        # 设置到基础配置
        keys = key.split('.')
        target = self._base

        for k in keys[:-1]:
            if isinstance(target, dict):
                if k not in target:
                    target[k] = {}
                target = target[k]
            elif hasattr(target, k):
                # dataclass, 获取属性
                target = getattr(target, k)
            else:
                # 创建新的字典
                new_target = {}
                setattr(target, k, new_target)
                target = new_target

        # 设置最终值
        if isinstance(target, dict):
            target[keys[-1]] = value
        else:
            setattr(target, keys[-1], value)

        logger.debug(f"配置已更新: {key} = {value}")

    def get_section(self, section: str) -> Any:
        """
        获取配置节

        Args:
            section: 节名称，如 "separators", "audio_utils"

        Returns:
            配置节对象
        """
        # 支持 dataclass 对象
        if hasattr(self._base, section):
            return getattr(self._base, section)
        if isinstance(self._base, dict) and section in self._base:
            return self._base[section]
        raise ConfigError(f"配置节不存在: {section}")

    def reset(self) -> None:
        """重置为默认配置"""
        self._base = copy.deepcopy(DEFAULT_CONFIG)
        self._user = {}
        logger.info("配置已重置为默认值")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # 支持 dataclass 对象
        if hasattr(self._base, 'to_dict') and callable(self._base.to_dict):
            return self._base.to_dict()
        if isinstance(self._base, dict):
            return copy.deepcopy(self._base)
        return _safe_asdict(self._base)

    def validate(self) -> bool:
        """
        验证配置有效性

        Returns:
            是否有效
        """
        errors = []

        # 验证设备配置
        device = self.get("separators.device")
        if device not in ["auto", "cpu", "cuda", "mps"]:
            errors.append(f"无效的设备类型: {device}")

        # 验证数值范围
        for key, min_val, max_val in [
            ("audio_utils.max_file_size_mb", 1, 10000),
            ("audio_utils.max_duration_seconds", 1, 86400),
            ("security.max_path_depth", 1, 100),
        ]:
            value = self.get(key)
            if value is not None and not (min_val <= value <= max_val):
                errors.append(f"{key} 值 {value} 超出范围 [{min_val}, {max_val}]")

        if errors:
            logger.error(f"配置验证失败: {errors}")
            return False

        return True

    @staticmethod
    def _get_nested(obj: Any, key: str) -> Optional[Any]:
        """获取嵌套属性，只支持 '.' 作为分隔符"""
        # 只使用 '.' 分隔符，不要替换 '_'
        keys = key.split('.')
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k)
            elif hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                return None
            if obj is None:
                return None
        return obj


@lru_cache(maxsize=1)
def get_config_path() -> Path:
    """
    获取配置路径

    优先级：
    1. 环境变量 SOMA_CONFIG_PATH
    2. ~/.soma/config.json
    """
    # 环境变量优先
    env_path = os.environ.get("SOMA_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser()

    # 默认路径
    default_path = Path("~/.soma/config.json").expanduser()
    return default_path


def get_config(
    path: Optional[Union[str, Path]] = None,
    auto_create: bool = True
) -> Config:
    """
    获取全局配置实例

    Args:
        path: 配置路径，默认使用 get_config_path()
        auto_create: 是否自动创建

    Returns:
        Config 实例
    """
    if path is None:
        path = get_config_path()
    return Config.load(path, auto_create=auto_create)


# 初始化配置模块
def init_config() -> Config:
    """初始化并返回全局配置"""
    config = get_config()

    # 确保必要的目录存在
    app_dir = Path(config.get("soma.app_dir", "~/.soma")).expanduser()
    for subdir in ["models", "cache", "temp", "logs", "workspace"]:
        (app_dir / subdir).mkdir(parents=True, exist_ok=True)

    return config
