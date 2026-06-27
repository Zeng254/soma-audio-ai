"""
Comparison page - Main page class.

Contains the ComparisonPage class definition with __init__, cleanup, and lifecycle methods.
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk
from concurrent.futures import ThreadPoolExecutor

from gui.pages.base import BasePage
from gui.pages.comparison.ui_mixin import ComparisonUIMixin
from gui.pages.comparison.worker_mixin import ComparisonWorkerMixin
from gui.pages.comparison.playback_mixin import ComparisonPlaybackMixin
from gui.utils import SettingsManager


class ComparisonPage(
    ComparisonUIMixin,
    ComparisonWorkerMixin,
    ComparisonPlaybackMixin,
    BasePage,
):
    """Comparison page for A/B testing voice conversion results.

    Uses Mixin pattern to split responsibilities:
    - ComparisonUIMixin: UI creation and layout
    - ComparisonWorkerMixin: Task management, execution, config, export
    - ComparisonPlaybackMixin: Audio playback and A/B comparison
    - BasePage: Base functionality (safe_after, widget_alive, etc.)
    """

    PAGE_NAME = "comparison"
    PAGE_TITLE = "Compare"
    PAGE_ICON = "\U0001f4ca"

    def __init__(self, parent: tk.Widget, **kwargs):
        """Initialize comparison page."""
        super().__init__(parent, **kwargs)

        # Settings manager (fix #1: singleton)
        self._settings = SettingsManager()

        # Task management (thread-safe, fix #6)
        self._tasks = []  # List of ComparisonTask dicts
        self._task_counter = 0
        self._tasks_lock = tk.BooleanVar()  # Placeholder, replaced below
        import threading
        self._tasks_lock = threading.Lock()
        self._tree_item_map = {}  # task_id -> treeview iid mapping (fix #4)

        # Model cache (fix #7)
        self._model_cache = []
        self._model_cache_time = 0

        # Processing state
        self._processing = False
        self._start_time = None
        self._elapsed_timer_id = None

        # Playback state (thread-safe, fix #3)
        self._current_player = None
        self._playback_lock = threading.Lock()

        # Thread pool (fix #6: configurable max_workers)
        max_workers = self._get_max_workers()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Tkinter variables
        self.source_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.pitch_shift = tk.IntVar(value=0)
        self.feature_extractor = tk.StringVar(value="hubert")
        self.f0_method = tk.StringVar(value="rmvpe")
        self.device = tk.StringVar(value="auto")
        self.output_sample_rate = tk.StringVar(value="40000")
        self.cluster_ratio = tk.DoubleVar(value=0.0)
        self.elapsed_var = tk.StringVar(value="0:00")

        # File info variables
        self.file_info_filename = tk.StringVar(value="No file selected")
        self.file_info_duration = tk.StringVar(value="--")
        self.file_info_samplerate = tk.StringVar(value="--")
        self.file_info_channels = tk.StringVar(value="--")
        self.file_info_filesize = tk.StringVar(value="--")

        # Volume
        self.volume_var = tk.DoubleVar(value=0.8)

        # Load settings
        self._last_directory = self._settings.get("comparison_last_dir", os.path.expanduser("~"))
        saved_output = self._settings.get("comparison_output_dir", "")
        if saved_output:
            self.output_dir.set(saved_output)

        # Build UI
        self._create_widgets()

    def cleanup(self):
        """Clean up resources (idempotent, fix #1)."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        # Stop playback
        self._stop_playback()

        # Cancel all running tasks
        with self._tasks_lock:
            for task in self._tasks:
                if task["status"] == "running":
                    task["cancel_flag"].set()

        # Shutdown thread pool (fix #1: explicit shutdown)
        try:
            if sys.version_info >= (3, 9):
                self._executor.shutdown(wait=False, cancel_futures=True)
            else:
                self._executor.shutdown(wait=False)
        except Exception:
            pass

        # Save settings
        if self.output_dir.get():
            self._settings.set("comparison_output_dir", self.output_dir.get())
        self._settings.set("comparison_last_dir", self._last_directory)
