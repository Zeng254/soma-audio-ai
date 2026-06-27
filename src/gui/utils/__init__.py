"""
GUI utility modules for SOMA.
"""

from gui.utils.settings_manager import SettingsManager
from gui.utils.common import open_folder, open_audio_file, widget_alive, safe_after
from gui.utils.constants import (
    # Status constants (fix #5)
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_DISPLAY,
    STATUS_READY,
    STATUS_STARTING,
    STATUS_LOADING_MODEL,
    STATUS_LOADING_AUDIO,
    STATUS_PROCESSING,
    STATUS_SAVING,
    STATUS_COMPLETED,
    STATUS_ERROR,
    STATUS_CANCELLED_UI,
    # TypedDict types (fix #3)
    TaskConfig,
    ComparisonTask,
    # GPU concurrency (fix #6)
    DEFAULT_MAX_WORKERS_CPU,
    DEFAULT_MAX_WORKERS_GPU,
    SETTING_KEY_MAX_WORKERS,
    SETTING_KEY_DEVICE_TYPE,
    # Feature / model constants
    FEATURE_EXTRACTORS,
    F0_METHODS,
    DEVICES,
    SAMPLE_RATES,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_F0_METHOD,
    DEFAULT_FEATURE_EXTRACTOR,
    DEFAULT_DEVICE,
    PITCH_MIN,
    PITCH_MAX,
    CLUSTER_RATIO_MIN,
    CLUSTER_RATIO_MAX,
    MODEL_CACHE_TTL,
    MODEL_SEARCH_MAX_DEPTH,
    AUDIO_EXTENSIONS,
    AUDIO_FILETYPES,
)

__all__ = [
    "SettingsManager",
    "open_folder",
    "open_audio_file",
    "widget_alive",
    "safe_after",
    # Status
    "STATUS_QUEUED", "STATUS_RUNNING", "STATUS_DONE", "STATUS_FAILED",
    "STATUS_CANCELLED", "STATUS_DISPLAY",
    "STATUS_READY", "STATUS_STARTING", "STATUS_LOADING_MODEL",
    "STATUS_LOADING_AUDIO", "STATUS_PROCESSING", "STATUS_SAVING",
    "STATUS_COMPLETED", "STATUS_ERROR", "STATUS_CANCELLED_UI",
    # Types
    "TaskConfig", "ComparisonTask",
    # GPU concurrency
    "DEFAULT_MAX_WORKERS_CPU", "DEFAULT_MAX_WORKERS_GPU",
    "SETTING_KEY_MAX_WORKERS", "SETTING_KEY_DEVICE_TYPE",
    # Feature / model
    "FEATURE_EXTRACTORS", "F0_METHODS", "DEVICES", "SAMPLE_RATES",
    "DEFAULT_SAMPLE_RATE", "DEFAULT_F0_METHOD", "DEFAULT_FEATURE_EXTRACTOR",
    "DEFAULT_DEVICE", "PITCH_MIN", "PITCH_MAX",
    "CLUSTER_RATIO_MIN", "CLUSTER_RATIO_MAX",
    "MODEL_CACHE_TTL", "MODEL_SEARCH_MAX_DEPTH",
    "AUDIO_EXTENSIONS", "AUDIO_FILETYPES",
]
