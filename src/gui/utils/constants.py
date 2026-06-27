"""
Shared parameter constants for SOMA GUI pages.

Centralizes option lists and defaults that are used across multiple pages
(InferencePage, ComparisonPage) to avoid duplication and ensure consistency.
"""

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
