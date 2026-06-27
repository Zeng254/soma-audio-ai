"""
Shared parameter constants for SOMA GUI pages.

Centralizes option lists, defaults, status strings, and type definitions
that are used across multiple pages to avoid duplication and ensure consistency.
"""

import threading
from typing import Optional, Dict, Any
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


# ── Task Status Constants (fix #5: status text constants) ──────────────

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

# Status display mapping (with icons)
STATUS_DISPLAY = {
    STATUS_QUEUED: "\u23f3 Queued",
    STATUS_RUNNING: "\u2699 Running...",
    STATUS_DONE: "\u2705 Done",
    STATUS_FAILED: "\u274c Failed",
    STATUS_CANCELLED: "\u23f9 Cancelled",
}

# UI status text constants (fix #5)
STATUS_READY = "Ready"
STATUS_STARTING = "Starting..."
STATUS_LOADING_MODEL = "Loading model..."
STATUS_LOADING_AUDIO = "Loading audio..."
STATUS_PROCESSING = "Processing..."
STATUS_SAVING = "Saving..."
STATUS_COMPLETED = "Completed"
STATUS_ERROR = "Error"
STATUS_CANCELLED_UI = "Cancelled"


# ── Task TypedDict (fix #3: type safety for task dicts) ────────────────

class TaskConfig(TypedDict):
    """Configuration for a single comparison task."""
    model: str
    pitch: int
    feature_extractor: str
    f0_method: str
    device: str
    sample_rate: str
    cluster_ratio: float


class ComparisonTask(TypedDict):
    """A comparison task with all metadata."""
    id: int
    config: TaskConfig
    status: str
    result_path: Optional[str]
    error: Optional[str]
    duration: Optional[float]
    cancel_flag: threading.Event
    uuid: str


# ── GPU Concurrency (fix #6: configurable) ─────────────────────────────

# Default max workers by device type
DEFAULT_MAX_WORKERS_CPU = 2
DEFAULT_MAX_WORKERS_GPU = 1

# SettingsManager keys
SETTING_KEY_MAX_WORKERS = "comparison_max_workers"
SETTING_KEY_DEVICE_TYPE = "comparison_device_type"


# ── Feature / Model Constants ──────────────────────────────────────────

# Feature extractors: name -> description
FEATURE_EXTRACTORS = {
    "hubert": "HuBERT Base (default, good quality)",
    "contentvec": "ContentVec (alternative features)",
}

# F0 extraction methods: name -> description
F0_METHODS = {
    "dio": "DIO (fast, default)",
    "harvest": "Harvest (better quality, slower)",
    "rmvpe": "RMVPE (deep learning, best quality)",
    "crepe": "Crepe (neural pitch, high quality)",
}

# Device options: name -> description
DEVICES = {
    "auto": "Auto-detect (GPU if available)",
    "cpu": "CPU (always available)",
    "cuda": "CUDA (NVIDIA GPU)",
}

# Available output sample rates (Hz)
SAMPLE_RATES = ["16000", "32000", "40000", "44100", "48000"]

# Default sample rate
DEFAULT_SAMPLE_RATE = "40000"

# Default F0 method
DEFAULT_F0_METHOD = "dio"

# Default feature extractor
DEFAULT_FEATURE_EXTRACTOR = "hubert"

# Default device
DEFAULT_DEVICE = "auto"

# Pitch shift range
PITCH_MIN = -12
PITCH_MAX = 12

# Cluster ratio range
CLUSTER_RATIO_MIN = 0.0
CLUSTER_RATIO_MAX = 1.0

# Model cache TTL in seconds
MODEL_CACHE_TTL = 10.0

# Model search max depth
MODEL_SEARCH_MAX_DEPTH = 3

# Supported audio file extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# Audio file dialog filter patterns
AUDIO_FILETYPES = [
    ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"),
    ("WAV files", "*.wav"),
    ("MP3 files", "*.mp3"),
    ("FLAC files", "*.flac"),
    ("All files", "*.*"),
]
