"""
SOMA Security module - Path security validation

Provides path security checks，Prevents path traversal attacks（Path Traversal）。

Attack example:
- ``../../etc/passwd``
- ``/home/user/../root/.ssh/id_rsa``
- ``\\\\UNC\\path\\to\\share``

Usage:
    from src.security.path_validator import PathValidator, safe_path

    # Validate path
    validator = PathValidator(allowed_dirs=["/home/user/projects"])
    validator.validate("/home/user/projects/audio.wav")  # OK
    validator.validate("/tmp/evil.wav")  # Raises PathTraversalError

    # Use convenience function
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
    Path security validator

    Features:
    - Detect path traversal attacks (.. components)
    - Validate path is within allowed directory
    - Normalize path (process symlinks, relative paths)
    - CheckPathDepth

    Usage example:
        validator = PathValidator(
            allowed_dirs=["/data/audio", "~/.soma"],
            allow_symlinks=False,
            max_depth=10
        )

        # Validate single path
        validator.validate("/data/audio/sample.wav")

        # SecurityParsePath
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
        Initialize path validator

        Args:
            allowed_dirs: Allowed base directory list
            allow_symlinks: Whether to allow symlinks
            max_depth: MaximumPathDepth
            defaults: Security default configuration
        """
        self.defaults = defaults or SecurityDefaults()

        # Parse allowed directories
        self.allowed_dirs: List[Path] = []
        dirs = allowed_dirs or self.defaults.allowed_base_dirs
        for d in dirs:
            p = Path(d).expanduser().resolve()
            if not p.exists():
                logger.warning(f"Allowed directory does not exist, will be created: {p}")
                p.mkdir(parents=True, exist_ok=True)
            self.allowed_dirs.append(p)

        self.allow_symlinks = allow_symlinks
        self.max_depth = max_depth or self.defaults.max_path_depth

    def validate(self, path: Union[str, Path]) -> Path:
        """
        Validate path security

        Args:
            path: Path to validate

        Returns:
            Secure parsed path object

        Raises:
            PathTraversalError: Detected path traversal attack
            ValueError: Path invalid
        """
        # 1. Check empty path
        if not str(path).strip():
            raise ValueError("Path cannot be empty")

        path_str = str(path)

        # 2. Check UNC path (Windows attack vector)
        path_str = str(path)
        if path_str.startswith('\\\\') or path_str.startswith('//'):
            raise PathTraversalError(
                attempted_path=path_str,
                allowed_base=None,
            )

        # 3. Convert to absolute path and normalize
        # Always resolve to get the canonical path
        # For non-existent paths, resolve() still normalizes the path
        path_obj = Path(path)
        try:
            resolved = path_obj.expanduser().resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve path: {path} - {e}")

        # 4. Check resolved path for traversal attempts
        # This single check after resolve() is sufficient since resolve() normalizes the path
        if '..' in resolved.parts:
            raise PathTraversalError(
                attempted_path=str(resolved),
                allowed_base=None,
            )

        # 6. Check symlinks - detect and reject symlinks that escape allowed dirs
        try:
            if not self.allow_symlinks:
                # Check if any component of the path is a symlink
                current = Path(resolved)
                while current != current.parent:
                    if current.is_symlink():
                        # Resolve the symlink target and check if it's in allowed dirs
                        link_target = current.resolve()
                        if not self._is_in_allowed_dirs(link_target):
                            raise PathTraversalError(
                                attempted_path=str(path),
                                allowed_base=str(self.allowed_dirs[0]) if self.allowed_dirs else None,
                                message=f"Symlink at {current} points outside allowed directories",
                            )
                    current = current.parent
        except (OSError, ValueError) as e:
            # If we can't check symlinks, log and continue
            logger.warning(f"Could not check symlinks for path {path}: {e}")

        # 7. CheckPathDepth
        self._check_depth(resolved)

        # 8. Check if within allowed directory range
        if not self._is_in_allowed_dirs(resolved):
            raise PathTraversalError(
                attempted_path=str(path),
                allowed_base=str(self.allowed_dirs[0]) if self.allowed_dirs else None,
            )

        logger.debug(f"Path validation passed: {path} -> {resolved}")
        return resolved

    def resolve(self, path: Union[str, Path]) -> Path:
        """
        Secure parse path (does not check existence)

        Args:
            path: Path to parse

        Returns:
            Parsed path object

        Raises:
            PathTraversalError: Detected path traversal attack
        """
        if not str(path).strip():
            raise ValueError("Path cannot be empty")

        path_str = str(path)

        # CheckPathIterate
        normalized = os.path.normpath(path_str)
        if '..' in Path(normalized).parts:
            raise PathTraversalError(
                attempted_path=path_str,
                allowed_base=None,
            )

        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot parse path: {path} - {e}")

        return resolved

    def _is_in_allowed_dirs(self, path: Path) -> bool:
        """Check if path is within allowed directory range using is_relative_to()"""
        # If no allowed directories configured, allow absolute paths
        if not self.allowed_dirs:
            return path.is_absolute()
        
        # Use is_relative_to() for robust path containment check (Python 3.9+)
        # This is more secure than string matching as it handles edge cases correctly
        for allowed_dir in self.allowed_dirs:
            try:
                if path.is_relative_to(allowed_dir):
                    return True
            except (ValueError, TypeError):
                # Fallback for older Python versions or edge cases
                try:
                    path.relative_to(allowed_dir)
                    return True
                except ValueError:
                    continue
        return False

    def _check_depth(self, path: Path) -> None:
        """CheckPathDepth"""
        depth = len(path.parts)
        if depth > self.max_depth:
            raise PathTraversalError(
                attempted_path=str(path),
                allowed_base=None,
            )

    def is_safe(self, path: Union[str, Path]) -> bool:
        """
        Check if path is secure (does not raise exception)

        Args:
            path: Path to check

        Returns:
            bool: Whether secure
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
    Secure get path convenience function

    Args:
        path: Input path
        base_dir: Allowed base directory

    Returns:
        Validated path

    Raises:
        PathTraversalError: Path not secure
    """
    if base_dir:
        validator = PathValidator(allowed_dirs=[base_dir])
    else:
        validator = PathValidator()
    return validator.validate(path)


def safe_join(*parts: str) -> str:
    """
    Securely concatenate path components

    Args:
        *parts: Path components

    Returns:
        Concatenated path

    Raises:
        PathTraversalError: Path traversal attack
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
    Ensure directory exists, create if not exists

    Args:
        path: Directory path
        mode: DirectoryPermission

    Returns:
        Path Object

    Raises:
        PathTraversalError: Path not secure
    """
    p = Path(path).expanduser().resolve()
    # Validate path security
    normalized = os.path.normpath(str(p))
    if '..' in Path(normalized).parts:
        raise PathTraversalError(
            attempted_path=str(path),
            allowed_base=None,
        )
    p.mkdir(parents=True, exist_ok=True, mode=mode)
    return p


# Global validator instance
_default_validator: Optional[PathValidator] = None


def get_validator() -> PathValidator:
    """
    Get global PathValidator instance

    Returns:
        PathValidator Instance
    """
    global _default_validator
    if _default_validator is None:
        _default_validator = PathValidator()
    return _default_validator
