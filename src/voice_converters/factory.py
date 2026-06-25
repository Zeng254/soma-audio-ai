"""
Voice Converter Factory - Voice conversion engine factory

Provides unified engine creation and management interface.
Supports automatic model type detection.
"""

from typing import Optional, Dict, List, Type, Any, Union
from pathlib import Path
import json
import os

from .base import (
    BaseVoiceConverter,
    ConverterType,
    ConversionParams,
    ModelInfo,
)


class ConverterFactory:
    """
    Voice converter factory
    
    Responsible for:
    - Automatic model type detection
    - Create appropriate converter instance
    - Manage converter lifecycle
    - Cache created converters
    """
    
    # Engine registry
    _engines: Dict[ConverterType, Type[BaseVoiceConverter]] = {}
    
    # Model type identifier
    MODEL_TYPE_INDICATORS = {
        # RVC identifier
        ".pth": "rvc",
        "rvc": ["model", "emb", "f0"],
        
        # SoVITS identifier
        "G_*.pth": "sovits",
        "sovits": ["config", "mel", "spk"],
    }
    
    @classmethod
    def register_engine(
        cls,
        converter_type: ConverterType,
        engine_class: Type[BaseVoiceConverter]
    ):
        """
        Register voice conversion engine
        
        Args:
            converter_type: Engine class type
            engine_class: Engine class
        """
        cls._engines[converter_type] = engine_class
    
    @classmethod
    def create_converter(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
        index_path: Optional[str] = None,
        engine: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseVoiceConverter:
        """
        CreateVoice converter
        
        Automatically detect model type and create corresponding converter.
        
        Args:
            model_path: ModelFile path
            config_path: Configuration filePath
            index_path: IndexFile path
            engine: Force specify engine ('rvc', 'sovits')
            device: Run device
            **kwargs: OtherParameter
            
        Returns:
            BaseVoiceConverter: Converter instance
            
        Raises:
            FileNotFoundError: Model file does not exist
            ValueError: Cannot detect model class type
        """
        model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # If config_path not provided, try to find
        if config_path is None:
            config_path = cls._find_config(model_file)
        
        # Detect model class type
        if engine is None:
            engine = cls.identify_model_type(
                model_path,
                config_path,
                index_path
            )
        
        # Get engine class type
        try:
            converter_type = ConverterType(engine.lower())
        except ValueError:
            raise ValueError(
                f"Unknown engine type: {engine}. "
                f"Supported: {', '.join([e.value for e in ConverterType])}"
            )
        
        # Create converter
        converter = cls._create_converter_instance(
            converter_type,
            device,
            **kwargs
        )
        
        # LoadModel
        converter.load_model(
            model_path,
            config_path=config_path,
            index_path=index_path,
            **kwargs
        )
        
        return converter
    
    @classmethod
    def _create_converter_instance(
        cls,
        converter_type: ConverterType,
        device: Optional[str],
        **kwargs
    ) -> BaseVoiceConverter:
        """
        Create converter instance
        
        Args:
            converter_type: Converter class type
            device: Device
            **kwargs: OtherParameter
            
        Returns:
            BaseVoiceConverter: Converter instance
        """
        # Check if already registered
        if converter_type in cls._engines:
            engine_class = cls._engines[converter_type]
        else:
            # Dynamic import
            if converter_type == ConverterType.RVC:
                from .rvc_converter import RVCConverter
                engine_class = RVCConverter
            elif converter_type == ConverterType.SOVITS:
                from .sovits_converter import SoVITSConverter
                engine_class = SoVITSConverter
            else:
                raise ValueError(f"Unsupported converter type: {converter_type}")
        
        # CreateInstance
        return engine_class(device=device, **kwargs)
    
    @classmethod
    def identify_model_type(
        cls,
        model_path: str,
        config_path: Optional[str] = None,
        index_path: Optional[str] = None,
    ) -> str:
        """
        Detect model class type
        
        Args:
            model_path: ModelPath
            config_path: Configuration filePath
            index_path: IndexFile path
            
        Returns:
            str: Model class type ('rvc' or 'sovits')
        """
        model_file = Path(model_path)
        
        # 1. Check from filename
        model_name = model_file.name.lower()
        
        if model_name.startswith("G_") or model_name.startswith("D_"):
            # SoVITS naming convention
            return "sovits"
        
        # 2. FromConfigurationCheck
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # SoVITS ConfigurationFeature
                if "train" in config or "model" in config:
                    if "spk" in config or "n_speakers" in config:
                        return "sovits"
                
                # RVC ConfigurationFeature
                if "emb" in config or "f0" in config:
                    return "rvc"
                    
            except Exception:
                pass
        
        # 3. FromIndexFileCheck
        if index_path and Path(index_path).exists():
            index_file = Path(index_path)
            if index_file.suffix == ".index":
                return "rvc"
        
        # 4. Check configuration in same directory
        possible_configs = [
            model_file.parent / "config.json",
            model_file.parent / "configs" / "config.json",
            model_file.parent.parent / "config.json",
        ]
        
        for config_file in possible_configs:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    if "train" in config:
                        return "sovits"
                    elif "emb" in config or "f0" in config:
                        return "rvc"
                except Exception:
                    continue
        
        # 5. Default values (prefer RVC)
        return "rvc"
    
    @classmethod
    def _find_config(cls, model_file: Path) -> Optional[str]:
        """FindConfiguration file"""
        # SoVITS ConfigurationFindPath
        sovits_paths = [
            model_file.parent / "config.json",
            model_file.parent / "configs" / "config.json",
            model_file.parent / "sovits_config.json",
            model_file.parent.parent / "config.json",
        ]
        
        for path in sovits_paths:
            if path.exists():
                return str(path)
        
        # RVC does not need configuration file (optional)
        return None
    
    @classmethod
    def get_available_engines(cls) -> List[Dict[str, Any]]:
        """
        Get available engine list
        
        Returns:
            List[Dict]: Engine info list
        """
        engines = []
        
        # Check RVC
        try:
            from .rvc_converter import RVCConverter
            if RVCConverter.is_available():
                engines.append({
                    "type": "rvc",
                    "name": RVCConverter.get_engine_name(),
                    "supported_formats": RVCConverter.get_supported_formats(),
                    "available": True,
                })
        except ImportError:
            engines.append({
                "type": "rvc",
                "name": "RVC v2",
                "available": False,
                "error": "torch not installed",
            })
        
        # Check SoVITS
        try:
            from .sovits_converter import SoVITSConverter
            if SoVITSConverter.is_available():
                engines.append({
                    "type": "sovits",
                    "name": SoVITSConverter.get_engine_name(),
                    "supported_formats": SoVITSConverter.get_supported_formats(),
                    "available": True,
                })
        except ImportError:
            engines.append({
                "type": "sovits",
                "name": "So-VITS-SVC 4.1",
                "available": False,
                "error": "torch not installed",
            })
        
        return engines
    
    @classmethod
    def get_recommended_params(cls, engine: str) -> ConversionParams:
        """
        Get engine recommended parameters
        
        Args:
            engine: Engine class type
            
        Returns:
            ConversionParams: Recommended parameters
        """
        if engine == "rvc":
            return ConversionParams(
                pitch_shift=0,
                pitch_algo="rmvpe",
                vpm=0.5,
                timbre_protection=0.5,
                rms_mix=0.5,
            )
        elif engine == "sovits":
            return ConversionParams(
                pitch_shift=0,
                pitch_algo="dio",
                vpm=0.5,
                timbre_protection=0.5,
                rms_mix=0.5,
            )
        else:
            return ConversionParams()


class VoiceConverterManager:
    """
    Voice converter manager
    
    Lifecycle management and resource sharing
    """
    
    def __init__(self):
        """Initialize manager"""
        self._converters: Dict[str, BaseVoiceConverter] = {}
        self._current: Optional[str] = None
    
    def load(
        self,
        name: str,
        model_path: str,
        **kwargs
    ) -> BaseVoiceConverter:
        """
        Load converter
        
        Args:
            name: Converter name
            model_path: ModelPath
            **kwargs: OtherParameter
            
        Returns:
            BaseVoiceConverter: Converter instance
        """
        # If already exists, uninstall first
        if name in self._converters:
            self.unload(name)
        
        # Create new converter
        converter = ConverterFactory.create_converter(
            model_path,
            **kwargs
        )
        
        self._converters[name] = converter
        self._current = name
        
        return converter
    
    def get(self, name: Optional[str] = None) -> Optional[BaseVoiceConverter]:
        """
        Get converter
        
        Args:
            name: Converter name, None returns current
            
        Returns:
            BaseVoiceConverter or None
        """
        if name is None:
            name = self._current
        
        return self._converters.get(name)
    
    def unload(self, name: str):
        """
        Unload converter
        
        Args:
            name: Converter name
        """
        if name in self._converters:
            converter = self._converters[name]
            converter.unload()
            del self._converters[name]
            
            if self._current == name:
                # Select another as current
                self._current = next(iter(self._converters.keys()), None)
    
    def unload_all(self):
        """Unload all converters"""
        for converter in self._converters.values():
            converter.unload()
        self._converters.clear()
        self._current = None
    
    def list_converters(self) -> List[str]:
        """List loaded converters"""
        return list(self._converters.keys())
    
    @property
    def current(self) -> Optional[BaseVoiceConverter]:
        """Get current converter"""
        return self.get()
    
    def __enter__(self) -> "VoiceConverterManager":
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.unload_all()
        return False


# Register default engines (lazy import, log on failure but do not crash)
import logging

logger = logging.getLogger(__name__)

try:
    from .rvc_converter import RVCConverter
    from .sovits_converter import SoVITSConverter
    
    ConverterFactory.register_engine(ConverterType.RVC, RVCConverter)
    ConverterFactory.register_engine(ConverterType.SOVITS, SoVITSConverter)
except ImportError as e:
    logger.warning(
        f"Failed to register default voice converter engines: {e}. "
        "Some features may not be available."
    )
