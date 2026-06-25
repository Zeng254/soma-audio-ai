"""
SOMA 安全模块 - 路径安全校验

提供路径安全检查，防止路径遍历攻击（Path Traversal）。

攻击示例:
- ``../../etc/passwd``
- ``/home/user/../root/.ssh/id_rsa``
- ``\\\\UNC\\path\\to\\share``

使用方式:
    from src.security.path_validator import PathValidator, safe_path

    # 验证路径
    validator = PathValidator(allowed_dirs=["/home/user/projects"])
    validator.validate("/home/user/projects/audio.wav")  # OK
    validator.validate("/tmp/evil.wav")  # 抛出 PathTraversalError

    # 使用便捷函数
    safe_path("~/projects/../secrets.txt", base_dir="~/projects")
"""

import os
import logging
from pathlib import Path
from typing import List, Union, Optional
from dataclasses import dataclass, field

from src.config import SecurityDefaults
from src.exceptions import SecurityError, PathTraversalError

logger = logging.getLogger(__name__)


class PathValidator:
    """
    路径安全校验器

    功能:
    - 检测路径遍历攻击 (.. 组件)
    - 验证路径在允许的目录范围内
    - 规范化路径（处理符号链接、相对路径）
    - 检查路径深度

    使用示例:
        validator = PathValidator(
            allowed_dirs=["/data/audio", "~/.soma"],
            allow_symlinks=False,
            max_depth=10
        )

        # 验证单个路径
        validator.validate("/data/audio/sample.wav")

        # 安全解析路径
        safe_path = validator.resolve("~/data/../data/secret.wav")
    """

    def __init__(
        self,
        allowed_dirs: Optional[List[Union[str, Path]]] = None,
        allow_symlinks: bool = False,
        max_depth: int = 20,
        defaults: Optional[SecurityDefaults] = None
    ):
        """
        初始化路径验证器

        Args:
            allowed_dirs: 允许的基础目录列表
            allow_symlinks: 是否允许符号链接
            max_depth: 最大路径深度
            defaults: 安全默认配置
        """
        self.defaults = defaults or SecurityDefaults()

        # 解析允许的目录
        self.allowed_dirs: List[Path] = []
        dirs = allowed_dirs or self.defaults.allowed_base_dirs
        for d in dirs:
            p = Path(d).expanduser().resolve()
            if not p.exists():
                logger.warning(f"允许的目录不存在，将被创建: {p}")
                p.mkdir(parents=True, exist_ok=True)
            self.allowed_dirs.append(p)

        self.allow_symlinks = allow_symlinks
        self.max_depth = max_depth or self.defaults.max_path_depth

    def validate(self, path: Union[str, Path]) -> Path:
        """
        验证路径安全性

        Args:
            path: 待验证的路径

        Returns:
            安全解析后的 Path 对象

        Raises:
            PathTraversalError: 检测到路径遍历攻击
            ValueError: 路径无效
        """
        # 1. 检查空路径
        if not str(path).strip():
            raise ValueError("路径不能为空")

        path_str = str(path)

        # 2. 检查路径遍历攻击（使用 normpath 规范化后检查 .. 组件）
        normalized = os.path.normpath(path_str)
        if '..' in Path(normalized).parts:
            raise PathTraversalError(
                attempted_path=path_str,
                allowed_base=None,
            )

        # 3. 检查 UNC 路径（Windows 攻击向量）
        if normalized.startswith('\\\\') or normalized.startswith('//'):
            raise PathTraversalError(
                attempted_path=path_str,
                allowed_base=None,
            )

        # 4. 转换为绝对路径并规范化
        path_obj = Path(path)
        try:
            if path_obj.exists():
                resolved = path_obj.expanduser().resolve()
            else:
                # 路径不存在，只进行基本解析
                resolved = path_obj.expanduser().resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"无法解析路径: {path} - {e}")

        # 5. 再次检查规范化后的路径
        normalized_resolved = os.path.normpath(str(resolved))
        if '..' in Path(normalized_resolved).parts:
            raise PathTraversalError(
                attempted_path=str(resolved),
                allowed_base=None,
            )

        # 6. 检查符号链接
        try:
            if not self.allow_symlinks and path_obj.is_symlink():
                raise PathTraversalError(
                    attempted_path=str(path),
                    allowed_base=None,
                )
        except (OSError, ValueError):
            pass  # 文件不存在时跳过符号链接检查

        # 7. 检查路径深度
        self._check_depth(resolved)

        # 8. 检查是否在允许的目录范围内
        if not self._is_in_allowed_dirs(resolved):
            raise PathTraversalError(
                attempted_path=str(path),
                allowed_base=str(self.allowed_dirs[0]) if self.allowed_dirs else None,
            )

        logger.debug(f"路径验证通过: {path} -> {resolved}")
        return resolved

    def resolve(self, path: Union[str, Path]) -> Path:
        """
        安全解析路径（不检查存在性）

        Args:
            path: 待解析的路径

        Returns:
            解析后的 Path 对象

        Raises:
            PathTraversalError: 检测到路径遍历攻击
        """
        if not str(path).strip():
            raise ValueError("路径不能为空")

        path_str = str(path)

        # 检查路径遍历
        normalized = os.path.normpath(path_str)
        if '..' in Path(normalized).parts:
            raise PathTraversalError(
                attempted_path=path_str,
                allowed_base=None,
            )

        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"无法解析路径: {path} - {e}")

        return resolved

    def _is_in_allowed_dirs(self, path: Path) -> bool:
        """检查路径是否在允许的目录范围内"""
        # 如果没有配置允许目录，允许绝对路径
        if not self.allowed_dirs:
            return path.is_absolute()
        
        # 检查是否在允许的目录内
        for allowed_dir in self.allowed_dirs:
            try:
                path.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
        return False

    def _check_depth(self, path: Path) -> None:
        """检查路径深度"""
        depth = len(path.parts)
        if depth > self.max_depth:
            raise PathTraversalError(
                attempted_path=str(path),
                allowed_base=None,
            )

    def is_safe(self, path: Union[str, Path]) -> bool:
        """
        检查路径是否安全（不抛出异常）

        Args:
            path: 待检查的路径

        Returns:
            bool: 是否安全
        """
        try:
            self.validate(path)
            return True
        except (SecurityError, ValueError):
            return False


def safe_path(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    安全获取路径的便捷函数

    Args:
        path: 输入路径
        base_dir: 允许的基准目录

    Returns:
        验证后的路径

    Raises:
        PathTraversalError: 路径不安全
    """
    if base_dir:
        validator = PathValidator(allowed_dirs=[base_dir])
    else:
        validator = PathValidator()
    return validator.validate(path)


def safe_join(*parts: str) -> str:
    """
    安全地拼接路径组件

    Args:
        *parts: 路径组件

    Returns:
        拼接后的路径

    Raises:
        PathTraversalError: 路径遍历攻击
    """
    path = os.path.join(*parts)
    normalized = os.path.normpath(path)
    if '..' in Path(normalized).parts:
        raise PathTraversalError(
            attempted_path=path,
            allowed_base=None,
        )
    return path


def ensure_directory(path: Union[str, Path], mode: int = 0o755) -> Path:
    """
    确保目录存在，如不存在则创建

    Args:
        path: 目录路径
        mode: 目录权限

    Returns:
        Path 对象

    Raises:
        PathTraversalError: 路径不安全
    """
    p = Path(path).expanduser().resolve()
    # 验证路径安全性
    normalized = os.path.normpath(str(p))
    if '..' in Path(normalized).parts:
        raise PathTraversalError(
            attempted_path=str(path),
            allowed_base=None,
        )
    p.mkdir(parents=True, exist_ok=True, mode=mode)
    return p


# 全局验证器实例
_default_validator: Optional[PathValidator] = None


def get_validator() -> PathValidator:
    """
    获取全局 PathValidator 实例

    Returns:
        PathValidator 实例
    """
    global _default_validator
    if _default_validator is None:
        _default_validator = PathValidator()
    return _default_validator
