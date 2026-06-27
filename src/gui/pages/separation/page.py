"""
SeparationPage - main page class for audio source separation.

Combines SeparationUIMixin and SeparationWorkerMixin to provide
the complete separation page functionality.

MRO (Method Resolution Order):
    SeparationPage -> BasePage -> SeparationUIMixin -> SeparationWorkerMixin -> object
    - BasePage provides: safe_after, _widget_alive, cleanup, on_show, on_hide
    - SeparationUIMixin provides: _create_widgets, _create_*_section, _browse_*, file info
    - SeparationWorkerMixin provides: _start_separation, _stop_separation, _separation_worker,
      _separation_complete, _separation_error, timer, logging
    - No method name conflicts between mixins (verified).
"""

import tkinter as tk
import threading
import os
from typing import Optional, List

from gui.pages.base import BasePage
from gui.utils import SettingsManager

from .ui_mixin import SeparationUIMixin
from .worker_mixin import SeparationWorkerMixin


# MRO: SeparationPage -> BasePage -> SeparationUIMixin -> SeparationWorkerMixin -> object
class SeparationPage(BasePage, SeparationUIMixin, SeparationWorkerMixin):
    """
    Separation page for audio source separation.

    Features:
    - Source audio selection with file info display and drop zone
    - Separation mode: 2-stem, 4-stem, HPSS, Dereverb
    - Backend selection: librosa (default), demucs, HPSS
    - Dereverb toggle
    - Output format: wav, mp3, flac
    - Output directory selection with directory memory
    - Progress bar with status text and elapsed time
    - Completion dialog with open folder button
    - Friendly error handling
    """

    PAGE_NAME = "Separation"
    PAGE_ICON = "\U0001f3bc"
    PAGE_DESCRIPTION = "Separate audio tracks"

    # Separation modes
    MODES = {
        "2-stem (Vocals + Accompaniment)": "2stems",
        "4-stem (Vocals + Drums + Bass + Other)": "4stems",
        "HPSS (Harmonic + Percussive)": "hpss",
        "Dereverb Only": "dereverb",
    }

    # Backends
    BACKENDS = {
        "librosa": "Lightweight, offline-first (default)",
        "demucs": "High quality deep learning model",
        "HPSS": "Spectral harmonic-percussive separation",
    }

    # Output formats
    OUTPUT_FORMATS = {
        "WAV": ".wav",
        "FLAC": ".flac",
        "MP3": ".mp3",
    }

    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the separation page.

        Cross-Mixin Attribute Contract:
        ================================
        All attributes below are shared between Mixins. Each Mixin reads/writes
        these attributes. This section serves as the contract between Mixins.

        UI Mixin reads/writes:
            - source_path, output_dir, separation_mode, backend, dereverb_enabled,
              output_format (Tkinter variables for UI controls)
            - progress_var, status_var, elapsed_var (progress display)
            - file_info_* (file info display variables)
            - _last_directory (remembered directory for file dialogs)

        Worker Mixin reads/writes:
            - _cancel_event (cancel signal, threading.Event)
            - _processing_thread (background thread reference)
            - _start_time, _elapsed_timer_id (elapsed time tracking)
            - source_path, output_dir, separation_mode, backend, dereverb_enabled,
              output_format (read parameters for separation)
            - progress_var, status_var, elapsed_var (update progress display)
            - _last_directory (save directory preference)

        BasePage reads/writes:
            - _cleaned_up (idempotent cleanup flag)
        """
        super().__init__(parent, app)

        # ---- Settings manager (singleton, thread-safe) ----
        self._settings = SettingsManager()

        # ---- Cancel event (threading.Event for unified cancel mechanism) ----
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
        self.output_dir = tk.StringVar()

        # Separation parameters
        self.separation_mode = tk.StringVar(value="2-stem (Vocals + Accompaniment)")
        self.backend = tk.StringVar(value="librosa")
        self.dereverb_enabled = tk.BooleanVar(value=False)
        self.output_format = tk.StringVar(value="WAV")

        # Progress display
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        self.elapsed_var = tk.StringVar(value="")

        # File info variables
        self.file_info_duration = tk.StringVar(value="--")
        self.file_info_samplerate = tk.StringVar(value="--")
        self.file_info_channels = tk.StringVar(value="--")
        self.file_info_filesize = tk.StringVar(value="--")
        self.file_info_filename = tk.StringVar(value="No file selected")

        # ---- Remembered last directory (from SettingsManager) ----
        self._last_directory = self._settings.get(
            "separation_last_dir", os.path.expanduser("~")
        )

        # Output files
        self._output_files: List[str] = []
