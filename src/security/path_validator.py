"""
SOMA 安全模块 - 路径安全校验

提供路径安全检查，防止路径遍历攻击（Path Traversal Attack）。

攻击示例:
- ../../etc/passwd
- /home/user/../root/.ssh/id_rsa
- \\UNC\path\to\share

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

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """安全相关异常基类"""
    pass


class PathTraversalError(SecurityError):
    """路径遍历攻击检测异常"""
    def __init__(self, path: str, reason: str = ""):
        self.path = path
        self.reason = reason
        super().__init__(
            f"路径遍历攻击检测: {path!r} - {reason}"
        )


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
        path = Path(path)

        # 1. 检查空路径
        if not str(path).strip():
            raise ValueError("路径不能为空")

        # 2. 转换为绝对路径并规范化
        try:
            resolved = path.expanduser().resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"无法解析路径: {path} - {e}")

        # 3. 检查路径遍历攻击
        self._check_traversal(path, resolved)

        # 4. 检查符号链接
        if not self.allow_symlinks and path.exists() and path.is_symlink():
            raise PathTraversalError(
                str(path),
                "符号链接已被禁用"
            )

        # 5. 检查路径深度
        self._check_depth(resolved)

        # 6. 检查是否在允许的目录范围内
        if not self._is_in_allowed_dirs(resolved):
            raise PathTraversalError(
                str(path),
                f"路径不在允许的目录范围内: {[str(d) for d in self.allowed_dirs]}"
            )

        logger.debug(f"路径验证通过: {path} -> {resolved}")
        return resolved

    def _check_traversal(self, original: Path, resolved: Path) -> None:
        """检查路径遍历攻击"""
        original_str = str(original)

        # 检查明显的路径遍历模式
        if '..' in original_str:
            # 规范化路径不应该包含 ..
            parts = resolved.parts
            if '..' in parts:
                raise PathTraversalError(
                    original_str,
                    "检测到 .. 路径遍历"
                )

        # 检查是否在允许目录外结束
        for allowed_dir in self.allowed_dirs:
            try:
                resolved.relative_to(allowed_dir)
                return  # 路径在允许目录内
            except ValueError:
                continue

        # 检查最终解析路径是否在允许范围内
        is_allowed = any(
            str(resolved).startswith(str(d))
            for d in self.allowed_dirs
        )

        if not is_allowed:
            raise PathTraversalError(
                original_str,
                "解析后的路径不在允许范围内"
            )

    def _check_depth(self, path: Path) -> None:
        """检查路径深度"""
        depth = len(path.parts)
        if depth > self.max_depth:
            raise PathTraversalError(
                str(path),
                f"路径深度 {depth} 超过最大限制 {self.max_depth}"
            )

    def _is_in_allowed_dirs(self, path: Path) -> bool:
        """检查路径是否在允许的目录范围内"""
        for allowed_dir in self.allowed_dirs:
            try:
                path.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
        return False

    def resolve(self, path: Union[str, Path]) -> Path:
        """
        安全解析路径（验证后返回）

        与 validate() 的区别：
        - validate() 主要用于验证外部输入
        - resolve() 可以用于任何需要安全解析的场景

        Args:
            path: 待解析的路径

        Returns:
            安全解析后的路径
        """
        return self.validate(path)

    def is_safe(self, path: Union[str, Path]) -> bool:
        """
        检查路径是否安全（不抛出异常）

        Args:
            path: 待检查的路径

        Returns:
            是否安全
        """
        try:
            self.validate(path)
            return True
        except (SecurityError, ValueError):
            return False


# 全局默认验证器
_default_validator: Optional[PathValidator] = None


def get_validator() -> PathValidator:
    """获取全局路径验证器实例"""
    global _default_validator
    if _default_validator is None:
        _default_validator = PathValidator()
    return _default_validator


def safe_path(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    便捷的路径安全解析函数

    Args:
        path: 待解析的路径
        base_dir: 基础目录，如果提供则路径必须在此目录下

    Returns:
        安全解析后的路径

    Raises:
        PathTraversalError: 路径不安全

    示例:
        safe = safe_path("~/audio/../secrets.wav")
        safe = safe_path("../etc/passwd", base_dir="~/projects")
    """
    validator = get_validator()

    if base_dir:
        # 临时添加 base_dir 到允许列表
        original_dirs = validator.allowed_dirs.copy()
        try:
            base = Path(base_dir).expanduser().resolve()
            validator.allowed_dirs.append(base)
            return validator.validate(path)
        finally:
            validator.allowed_dirs = original_dirs
    else:
        return validator.validate(path)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    确保目录存在（安全版本）

    Args:
        path: 目录路径

    Returns:
        已创建的目录路径

    Raises:
        PathTraversalError: 路径不安全
    """
    safe = safe_path(path)

    if not safe.exists():
        safe.mkdir(parents=True, exist_ok=True)
        logger.debug(f"已创建目录: {safe}")

    return safe


def safe_join(*parts: str) -> Path:
    """
    安全地拼接路径组件

    Args:
        *parts: 路径组件

    Returns:
        拼接后的安全路径

    示例:
        safe = safe_join("~/data", "audio", "input.wav")
    """
    if not parts:
        raise ValueError("至少需要提供一个路径组件")

    # 第一个路径作为基础
    base = safe_path(parts[0])

    # 逐个添加后续组件
    for part in parts[1:]:
        # 清理路径中的 .. 和危险字符
        part = part.strip().replace('..', '').lstrip('/\\')

        if not part:
            continue

        base = base / part

    return safe_path(base)
