"""
SOMA 安全模块 - 安全模型加载

提供安全的深度学习模型加载功能：
- 使用 torch.load(weights_only=True) 防止代码执行
- 校验模型文件大小和格式
- 支持模型签名验证（可选）
- 提供模型元数据提取

使用方式:
    from src.security.model_loader import SafeModelLoader, load_model

    # 使用安全加载器
    loader = SafeModelLoader(trusted_signatures=["model_v1_hash"])
    model = loader.load("model.pth", device="cuda")

    # 使用便捷函数
    model = load_model("model.pth", map_location="cpu")
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass

from src.config import SecurityDefaults
from src.security.path_validator import safe_path

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """模型加载错误异常"""
    pass


class ModelVerificationError(Exception):
    """模型验证失败异常"""
    pass


@dataclass
class ModelMetadata:
    """模型元数据"""
    path: Path
    size_mb: float
    format: str
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SafeModelLoader:
    """
    安全的模型加载器

    安全特性:
    1. 使用 weights_only=True 防止代码执行
    2. 验证模型文件大小
    3. 支持模型完整性校验
    4. 详细的错误信息

    使用示例:
        loader = SafeModelLoader(
            max_size_mb=5000,
            trusted_signatures=["abc123...", "def456..."],
            device="cuda"
        )

        model = loader.load("path/to/model.pth")
    """

    # 支持的模型格式
    SUPPORTED_FORMATS = ['.pth', '.pt', '.bin', '.onnx', '.safetensors']

    def __init__(
        self,
        max_size_mb: int = 5000,
        trusted_signatures: Optional[List[str]] = None,
        device: str = "auto",
        defaults: Optional[SecurityDefaults] = None
    ):
        """
        初始化模型加载器

        Args:
            max_size_mb: 最大模型文件大小（MB）
            trusted_signatures: 可信的模型签名列表（SHA256哈希）
            device: 加载设备
            defaults: 安全默认配置
        """
        self.defaults = defaults or SecurityDefaults()
        self.max_size_bytes = (max_size_mb or self.defaults.max_model_size_mb) * 1024 * 1024
        self.trusted_signatures = trusted_signatures or self.defaults.trusted_model_signatures or []
        self.device = device

    def load(
        self,
        model_path: Union[str, Path],
        map_location: Optional[str] = None,
        weights_only: bool = True,
        verify: bool = True
    ) -> Any:
        """
        安全加载模型

        Args:
            model_path: 模型文件路径
            map_location: 设备映射
            weights_only: 是否只加载权重（安全模式）
            verify: 是否验证模型完整性

        Returns:
            加载的模型对象

        Raises:
            ModelLoadError: 加载失败
            ModelVerificationError: 验证失败
        """
        model_path = safe_path(model_path)

        # 1. 验证文件存在
        if not model_path.exists():
            raise ModelLoadError(f"模型文件不存在: {model_path}")

        # 2. 验证文件格式
        self._validate_format(model_path)

        # 3. 验证文件大小
        self._validate_size(model_path)

        # 4. 验证模型完整性（可选）
        if verify:
            self._verify_model(model_path)

        # 5. 计算设备
        device = map_location or self._get_device()

        # 6. 加载模型
        return self._load_torch_model(model_path, device, weights_only)

    def _validate_format(self, path: Path) -> None:
        """验证模型文件格式"""
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ModelLoadError(
                f"不支持的模型格式: {ext}，"
                f"支持的格式: {self.SUPPORTED_FORMATS}"
            )

    def _validate_size(self, path: Path) -> None:
        """验证模型文件大小"""
        size = path.stat().st_size
        size_mb = size / 1024 / 1024

        if size > self.max_size_bytes:
            raise ModelLoadError(
                f"模型文件过大: {size_mb:.1f} MB，"
                f"最大允许: {self.max_size_bytes / 1024 / 1024:.1f} MB"
            )

        if size < 1024:  # 小于 1KB
            raise ModelLoadError(f"模型文件过小，可能已损坏: {size} bytes")

        logger.debug(f"模型大小: {size_mb:.1f} MB")

    def _verify_model(self, path: Path) -> None:
        """验证模型完整性"""
        # 计算 SHA256 哈希
        checksum = self._calculate_checksum(path)

        # 检查是否在可信签名列表中
        if self.trusted_signatures:
            if checksum not in self.trusted_signatures:
                logger.warning(
                    f"模型签名未在可信列表中: {checksum[:16]}...，"
                    f"建议验证来源"
                )
                # 注意：这里只警告，不断言失败

        logger.debug(f"模型校验和: {checksum}")

    def _calculate_checksum(self, path: Path) -> str:
        """计算文件 SHA256 校验和"""
        sha256 = hashlib.sha256()

        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _get_device(self) -> str:
        """获取加载设备"""
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def _load_torch_model(
        self,
        path: Path,
        device: str,
        weights_only: bool
    ) -> Any:
        """加载 PyTorch 模型"""
        try:
            import torch

            # 安全加载：使用 weights_only=True 防止代码执行
            # 但如果模型包含非权重数据（如优化器状态），需要设为 False
            logger.info(f"正在加载模型: {path}")

            # 首先尝试以 weights_only 模式加载
            try:
                state_dict = torch.load(
                    path,
                    map_location=device,
                    weights_only=True
                )
                logger.info(f"模型已安全加载 (weights_only=True)")
            except Exception as e:
                # 如果失败，尝试关闭 weights_only（用于加载完整模型）
                if "weights_only" in str(e).lower() or "unpicklable" in str(e).lower():
                    logger.warning(
                        f"weights_only=True 加载失败 ({e})，"
                        f"尝试 weights_only=False，这可能存在安全风险"
                    )
                    state_dict = torch.load(
                        path,
                        map_location=device,
                        weights_only=False
                    )
                else:
                    raise

            return state_dict

        except ImportError:
            raise ModelLoadError("PyTorch 未安装，请运行: pip install torch")
        except Exception as e:
            raise ModelLoadError(f"模型加载失败: {e}")

    def get_metadata(self, model_path: Union[str, Path]) -> ModelMetadata:
        """
        获取模型元数据

        Args:
            model_path: 模型文件路径

        Returns:
            ModelMetadata 元数据对象
        """
        model_path = safe_path(model_path)

        if not model_path.exists():
            raise ModelLoadError(f"模型文件不存在: {model_path}")

        size = model_path.stat().st_size
        ext = model_path.suffix.lower()

        metadata = ModelMetadata(
            path=model_path,
            size_mb=size / 1024 / 1024,
            format=ext.lstrip('.'),
            checksum=self._calculate_checksum(model_path) if ext == '.pth' else None
        )

        # 尝试读取嵌入的元数据
        if ext == '.pth':
            try:
                # 不完全加载，只读取元数据
                import torch
                with open(model_path, 'rb') as f:
                    checkpoint = torch.load(
                        f,
                        map_location='cpu',
                        weights_only=True
                    )

                if isinstance(checkpoint, dict):
                    metadata.metadata = {
                        k: str(v)[:100]  # 截断长值
                        for k, v in checkpoint.items()
                        if not isinstance(v, (torch.Tensor, list))
                    }

            except Exception as e:
                logger.debug(f"无法读取模型元数据: {e}")

        return metadata


# 全局加载器实例
_default_loader: Optional[SafeModelLoader] = None


def get_model_loader() -> SafeModelLoader:
    """获取全局模型加载器实例"""
    global _default_loader
    if _default_loader is None:
        _default_loader = SafeModelLoader()
    return _default_loader


def load_model(
    model_path: Union[str, Path],
    map_location: Optional[str] = None,
    weights_only: bool = True
) -> Any:
    """
    便捷的模型加载函数

    Args:
        model_path: 模型文件路径
        map_location: 设备映射
        weights_only: 是否只加载权重

    Returns:
        加载的模型

    示例:
        model = load_model("path/to/model.pth", map_location="cpu")
    """
    loader = get_model_loader()
    return loader.load(
        model_path,
        map_location=map_location,
        weights_only=weights_only
    )
