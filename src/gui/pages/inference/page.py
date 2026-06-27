"""
Inference Page for SOMA GUI.

Provides interface for AI cover generation (voice conversion).
Uses Mixin pattern to separate UI creation, worker logic, and page lifecycle.
"""

import tkinter as tk
import threading
import os
from typing import Optional, List

from gui.pages.base import BasePage
from gui.utils import (
    SettingsManager,
    DEFAULT_SAMPLE_RATE, DEFAULT_F0_METHOD, DEFAULT_FEATURE_EXTRACTOR,
    DEFAULT_DEVICE,
)
from .ui_mixin import InferenceUIMixin
from .worker_mixin import InferenceWorkerMixin


class InferencePage(BasePage, InferenceUIMixin, InferenceWorkerMixin):
    """
    Inference page for generating AI covers.

    Features:
    - Source audio selection with file info and drop zone
    - Model selection with refresh
    - Pitch shift: +/-12 semitones
    - Feature extractor: hubert, contentvec
    - F0 method: dio, harvest, rmvpe, crepe
    - Device: auto / cpu / cuda
    - Sample rate selection
    - Clustering ratio slider
    - Multi-stage progress (load model -> feature extract -> vocoder)
    - Elapsed time display
    - Preprocessing options (separate vocals, dereverb)
    - Output path with auto-naming
    - Completion dialog with open folder button
    - Friendly error handling
    """

    PAGE_NAME = "Song Cover"
    PAGE_ICON = "\U0001f3b5"
    PAGE_DESCRIPTION = "Generate covers"

    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the inference page."""
        super().__init__(parent, app)

        # Settings manager (singleton, thread-safe)
        self._settings = SettingsManager()

        # Cancel event (threading.Event for unified cancel mechanism, fix #2)
        self._cancel_event = threading.Event()

        # State
        self._processing_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._elapsed_timer_id: Optional[str] = None

        # Variables
        self.source_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.pitch_shift = tk.IntVar(value=0)
        self.quality = tk.StringVar(value="High")

        # Advanced parameters
        self.feature_extractor = tk.StringVar(value=DEFAULT_FEATURE_EXTRACTOR)
        self.f0_method = tk.StringVar(value=DEFAULT_F0_METHOD)
        self.device = tk.StringVar(value=DEFAULT_DEVICE)
        self.output_sample_rate = tk.StringVar(value=DEFAULT_SAMPLE_RATE)
        self.cluster_ratio = tk.DoubleVar(value=0.0)

        # Preprocessing options
        self.separate_vocals = tk.BooleanVar(value=True)
        self.dereverb_audio = tk.BooleanVar(value=False)
        self.separation_mode = tk.StringVar(value="2-stem")

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        self.elapsed_var = tk.StringVar(value="")
        self.stage_var = tk.StringVar(value="")

        # File info variables
        self.file_info_duration = tk.StringVar(value="--")
        self.file_info_samplerate = tk.StringVar(value="--")
        self.file_info_channels = tk.StringVar(value="--")
        self.file_info_filesize = tk.StringVar(value="--")
        self.file_info_filename = tk.StringVar(value="No file selected")

        # Remembered last directory (from SettingsManager)
        self._last_directory = self._settings.get(
            "inference_last_dir", os.path.expanduser("~")
        )

        # Available models
        self._available_models: List[str] = []

        # Model scan cache (fix #7)
        self._model_cache: List[str] = []
        self._model_cache_time: float = 0.0

    def cleanup(self):
        """Clean up resources when page is destroyed."""
        super().cleanup()
        self._cancel_event.set()
        # Stop any running processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
