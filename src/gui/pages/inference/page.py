"""
Inference Page for SOMA GUI.

Provides interface for AI cover generation (voice conversion).
Uses Mixin pattern to separate UI creation, worker logic, and page lifecycle.

MRO (Method Resolution Order):
    InferencePage -> BasePage -> InferenceUIMixin -> InferenceWorkerMixin -> object
    - BasePage provides: safe_after, _widget_alive, cleanup, on_show, on_hide
    - InferenceUIMixin provides: _create_widgets, _create_*_section, _browse_*,
      file info, model management (_refresh_models, _find_model_file)
    - InferenceWorkerMixin provides: _start_conversion, _stop_conversion,
      _conversion_worker, _conversion_complete, _conversion_error,
      timer, logging, stage management
    - No method name conflicts between mixins (verified).
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


# MRO: InferencePage -> BasePage -> InferenceUIMixin -> InferenceWorkerMixin -> object
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
        """Initialize the inference page.

        Cross-Mixin Attribute Contract:
        ================================
        All attributes below are shared between Mixins. Each Mixin reads/writes
        these attributes. This section serves as the contract between Mixins.

        UI Mixin reads/writes:
            - source_path, output_path, selected_model (paths and model selection)
            - pitch_shift, quality, feature_extractor, f0_method, device,
              output_sample_rate, cluster_ratio (inference parameters)
            - separate_vocals, dereverb_audio, separation_mode (preprocessing)
            - progress_var, status_var, elapsed_var, stage_var (progress display)
            - file_info_* (file info display variables)
            - _last_directory (remembered directory for file dialogs)
            - _model_cache, _model_cache_time (model list cache)

        Worker Mixin reads/writes:
            - _cancel_event (cancel signal, threading.Event)
            - _processing_thread (background thread reference)
            - _start_time, _elapsed_timer_id (elapsed time tracking)
            - source_path, output_path, selected_model, pitch_shift,
              feature_extractor, f0_method, device, output_sample_rate,
              cluster_ratio, separate_vocals, dereverb_audio (read parameters)
            - progress_var, status_var, elapsed_var, stage_var (update progress)
            - _last_directory (save directory preference)

        BasePage reads/writes:
            - _cleaned_up (idempotent cleanup flag)
        """
        super().__init__(parent, app)

        # ---- Settings manager (singleton, thread-safe) ----
        self._settings = SettingsManager()

        # ---- Cancel event (threading.Event for unified cancel mechanism, fix #2) ----
        # Used by: WorkerMixin (set/clear), UI (check for stop button state)
        self._cancel_event = threading.Event()

        # ---- Processing state ----
        # Used by: WorkerMixin (thread management), UI (button enable/disable)
        self._processing_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._elapsed_timer_id: Optional[str] = None

        # ---- Tkinter variables (UI <-> Worker communication) ----
        # Source/output paths
        self.source_path = tk.StringVar()
        self.output_path = tk.StringVar()

        # Model selection
        self.selected_model = tk.StringVar()

        # Basic parameters
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

        # Progress display
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

        # ---- Remembered last directory (from SettingsManager) ----
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
