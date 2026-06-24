"""
SOMA GUI - 音频处理工作站图形界面

基于 PyQt6 实现的音频处理工作站界面
支持声音转换、音频分离、音效处理等功能
"""

from .main_window import MainWindow
from .workers import ConversionWorker
from .components import (
    AudioInputPanel,
    ModelConfigPanel,
    OutputPanel,
    StatusBar,
)

__version__ = "0.1.0"
__all__ = [
    "MainWindow",
    "ConversionWorker",
    "AudioInputPanel",
    "ModelConfigPanel",
    "OutputPanel",
    "StatusBar",
]
