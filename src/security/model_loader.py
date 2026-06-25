"""
SOMA Security module - SecurityModel loading

Provides secure deep learning model loading:
- Uses torch.load(weights_only=True) Prevents code execution
- Default strict mode, disables unsafe weights_only=False fallback
- Validate model file size and format
- Supports model signature validation (optional)
- Provides model metadata extraction

Usage:
    from src.security.model_loader import SafeModelLoader, load_model

    # Use secure loader (default strict mode)
    loader = SafeModelLoader(trusted_signatures=["model_v1_hash"])
    model = loader.load("model.pth", device="cuda")

    # If you really need to load model with non-weight data (e.g. optimizer state),
    # Explicitly set explicit_unsafe=True (at your own risk)
    loader = SafeModelLoader()
    model = loader.load("full_checkpoint.pth", explicit_unsafe=True)

    # Use convenience function
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
from src.exceptions import SOMAModelError, SOMAValidationError, SecurityError

logger = logging.getLogger(__name__)


# Compatibility alias - use unified exception system
class ModelLoadError(SOMAModelError):
    """Model loadingErrorException"""
    pass


class ModelVerificationError(SOMAValidationError):
    """Model validationFailException"""
    pass


class ModelSecurityError(SecurityError):
    """ModelSecurityCheckFailException"""
    pass


@dataclass
class ModelMetadata:
    """Model metadata"""
    path: Path
    size_mb: float
    format: str
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SafeModelLoader:
    """
    Secure model loader

    Security features:
    1. Default uses weights_only=True to prevent code execution
    2. Strict mode: default disables weights_only=False fallback
    3. ValidateModelFileSize
    4. Supports model integrity validation
    5. Detailed error info

    Usage example:
        loader = SafeModelLoader(
            max_size_mb=5000,
            trusted_signatures=["abc123...", "def456..."],
            device="cuda"
        )

        # Secure load (default)
        model = loader.load("path/to/model.pth")

        # Explicit fallback (requires user to confirm risk)
        model = loader.load("full_checkpoint.pth", explicit_unsafe=True)
    """

    # SupportsModelFormat
    SUPPORTED_FORMATS = ['.pth', '.pt', '.bin', '.onnx', '.safetensors']

    def __init__(
        self,
        max_size_mb: int = 5000,
        trusted_signatures: Optional[List[str]] = None,
        device: str = "auto",
        defaults: Optional[SecurityDefaults] = None
    ):
        """
        Initialize model loader

        Args:
            max_size_mb: MaximumModelFileSize（MB）
            trusted_signatures: Trusted model signature list (SHA256 hash)
            device: Load device
            defaults: Security default configuration
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
        verify: bool = True,
        explicit_unsafe: bool = False
    ) -> Any:
        """
        SecurityLoadModel

        Args:
            model_path: ModelFile path
            map_location: Device mapping
            weights_only: Whether to load only weights (security mode)
            verify: Whether to validate model integrity
            explicit_unsafe: Whether to explicitly allow unsafe mode (weights_only=False)
                            Default is False, strict mode

        Returns:
            LoadModelObject

        Raises:
            ModelLoadError: LoadFail
            ModelSecurityError: SecurityCheckFail
            ModelVerificationError: ValidateFail
        """
        try:
            model_path = safe_path(model_path)
        except Exception as e:
            if "Path not in allowed directory range" in str(e) or "does not exist" in str(e):
                raise ModelLoadError(f"Model path is not secure: {model_path}") from e
            raise

        # 1. Validate file exists
        if not model_path.exists():
            raise ModelLoadError(f"Model file does not exist: {model_path}")

        # 2. ValidateFileFormat
        self._validate_format(model_path)

        # 3. ValidateFileSize
        self._validate_size(model_path)

        # 4. Validate model integrity (optional)
        if verify:
            self._verify_model(model_path)

        # 5. Calculate device
        device = map_location or self._get_device()

        # 6. LoadModel
        return self._load_torch_model(
            model_path,
            device,
            weights_only=weights_only,
            explicit_unsafe=explicit_unsafe
        )

    def _validate_format(self, path: Path) -> None:
        """ValidateModelFileFormat"""
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ModelLoadError(
                f"Unsupported model format: {ext}, "
                f"SupportsFormat: {self.SUPPORTED_FORMATS}"
            )

    def _validate_size(self, path: Path) -> None:
        """ValidateModelFileSize"""
        size = path.stat().st_size
        size_mb = size / 1024 / 1024

        if size > self.max_size_bytes:
            raise ModelLoadError(
                f"Model file too large: {size_mb:.1f} MB, "
                f"Maximum allowed: {self.max_size_bytes / 1024 / 1024:.1f} MB"
            )

        if size < 1024:  # Less than 1KB
            raise ModelLoadError(f"Model file too small, possibly corrupted: {size} bytes")

        logger.debug(f"ModelSize: {size_mb:.1f} MB")

    def _verify_model(self, path: Path) -> None:
        """Validate model integrity"""
        # Calculate SHA256 hash
        checksum = self._calculate_checksum(path)

        # Check if in trusted signature list
        if self.trusted_signatures:
            if checksum not in self.trusted_signatures:
                logger.warning(
                    f"Model signature not in trusted list: {checksum[:16]}..., "
                    f"Recommend validating source"
                )
                # Note: Only warning here, does not assert failure

        logger.debug(f"Model validation sum: {checksum}")

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate file SHA256 validation sum"""
        sha256_hash = hashlib.sha256()

        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _get_device(self) -> str:
        """Get load device"""
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
        weights_only: bool = True,
        explicit_unsafe: bool = False
    ) -> Any:
        """
        Load PyTorch Model

        Args:
            path: ModelPath
            device: Target device
            weights_only: Whether to load only weights
            explicit_unsafe: Whether to explicitly allow unsafe mode

        Returns:
            Load model status dictionary or model object

        Raises:
            ModelLoadError: LoadFail
            ModelSecurityError: SecurityCheckFail
        """
        try:
            import torch

            logger.info(f"Loading model: {path} (weights_only={weights_only}, explicit_unsafe={explicit_unsafe})")

            # Strict mode: default uses weights_only=True
            if weights_only or not explicit_unsafe:
                # Security mode: only allow weights_only=True
                try:
                    state_dict = torch.load(
                        path,
                        map_location=device,
                        weights_only=True
                    )
                    logger.info(f"Model security loaded (weights_only=True)")
                    return state_dict
                except Exception as e:
                    error_msg = str(e).lower()
                    # Check if model contains data that cannot be loaded with weights_only
                    if "weights_only" in error_msg or "unpicklable" in error_msg or "storage" in error_msg:
                        if explicit_unsafe:
                            # User has explicitly confirmed risk, downgrading to insecure mode
                            logger.warning(
                                f"weights_only=True LoadFail ({e})，"
                                f"Since explicit_unsafe=True, will try weights_only=False"
                            )
                            state_dict = torch.load(
                                path,
                                map_location=device,
                                weights_only=False
                            )
                            logger.warning(f"Model loaded in insecure mode (weights_only=False), ensure model source is trusted")
                            return state_dict
                        else:
                            # Strict mode: raise exception
                            raise ModelSecurityError(
                                f"Model loading failed: model contains non-weight data (e.g. Python objects, lambda functions, etc.)，"
                                f"Cannot load in secure mode (weights_only=True)."
                                f"e.g. to load this class of model, use explicit_unsafe=True parameter, "
                                f"But ensure model source is trusted to avoid code execution risk."
                            )
                    else:
                        raise ModelLoadError(f"Model loadingFail: {e}")
            else:
                # Insecure mode (deprecated, only available when explicit_unsafe=True)
                state_dict = torch.load(
                    path,
                    map_location=device,
                    weights_only=False
                )
                logger.warning(f"Model loaded in insecure mode (weights_only=False), ensure model source is trusted")
                return state_dict

        except ImportError:
            raise ModelLoadError("PyTorch not installed, run: uv add torch torchaudio")
        except (ModelLoadError, ModelSecurityError):
            raise
        except Exception as e:
            raise ModelLoadError(f"Model loadingFail: {e}")

    def get_metadata(self, model_path: Union[str, Path]) -> ModelMetadata:
        """
        Get model metadata

        Args:
            model_path: ModelFile path

        Returns:
            ModelMetadata metadata object
        """
        model_path = safe_path(model_path)

        if not model_path.exists():
            raise ModelLoadError(f"Model file does not exist: {model_path}")

        size = model_path.stat().st_size
        ext = model_path.suffix.lower()

        metadata = ModelMetadata(
            path=model_path,
            size_mb=size / 1024 / 1024,
            format=ext.lstrip('.'),
            checksum=self._calculate_checksum(model_path) if ext == '.pth' else None
        )

        # Try to read embedded metadata
        if ext == '.pth':
            try:
                # Use weights_only=True to securely read metadata
                import torch
                with open(model_path, 'rb') as f:
                    checkpoint = torch.load(
                        f,
                        map_location='cpu',
                        weights_only=True
                    )

                if isinstance(checkpoint, dict):
                    metadata.metadata = {
                        k: str(v)[:100]  # Truncate long values
                        for k, v in checkpoint.items()
                        if not isinstance(v, (torch.Tensor, list))
                    }

            except Exception as e:
                logger.debug(f"Failed to read model metadata: {e}")

        return metadata


# Convenience function
def load_model(
    model_path: Union[str, Path],
    map_location: Optional[str] = None,
    explicit_unsafe: bool = False,
    **kwargs
) -> Any:
    """
    Convenient model loading function

    Args:
        model_path: ModelFile path
        map_location: Device mapping
        explicit_unsafe: Whether to explicitly allow unsafe mode
        **kwargs: OtherParameter

    Returns:
        LoadModelObject
    """
    loader = SafeModelLoader()
    return loader.load(
        model_path,
        map_location=map_location,
        explicit_unsafe=explicit_unsafe,
        **kwargs
    )


# Global loader instance
_default_loader: Optional[SafeModelLoader] = None


def get_model_loader() -> SafeModelLoader:
    """
    Get global SafeModelLoader instance

    Returns:
        SafeModelLoader Instance
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = SafeModelLoader()
    return _default_loader
