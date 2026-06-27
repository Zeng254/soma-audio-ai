"""
Comparison page - Main page class.

Contains the ComparisonPage class definition with __init__, cleanup, and lifecycle methods.

MRO (Method Resolution Order):
    ComparisonPage -> ComparisonUIMixin -> ComparisonWorkerMixin -> ComparisonPlaybackMixin -> BasePage -> object
    - BasePage provides: safe_after, _widget_alive, cleanup, on_show, on_hide
    - ComparisonUIMixin provides: _create_widgets, _create_*_section, _browse_*,
      file info, model management (_refresh_models, _find_model_file)
    - ComparisonWorkerMixin provides: task management (_add_task, _remove_selected_task,
      _clear_done_tasks), execution (_start_all_tasks, _run_task), config save/load,
      export, treeview updates
    - ComparisonPlaybackMixin provides: audio playback (_play_audio_file, _stop_playback),
      A/B comparison (_ab_switch_play, _play_b_after_a)
    - No method name conflicts between mixins (verified).
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


# MRO: ComparisonPage -> ComparisonUIMixin -> ComparisonWorkerMixin -> ComparisonPlaybackMixin -> BasePage -> object
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

    PAGE_NAME = "对比试听"
    PAGE_TITLE = "对比试听"
    PAGE_ICON = "\U0001f4ca"

    def __init__(self, parent: tk.Widget, **kwargs):
        """Initialize comparison page.

        Cross-Mixin Attribute Contract:
        ================================
        All attributes below are shared between Mixins. Each Mixin reads/writes
        these attributes. This section serves as the contract between Mixins.

        UI Mixin reads/writes:
            - source_path, output_dir, selected_model (paths and model selection)
            - pitch_shift, feature_extractor, f0_method, device, output_sample_rate,
              cluster_ratio (inference parameters for new tasks)
            - elapsed_var (elapsed time display)
            - file_info_* (file info display variables)
            - _last_directory (remembered directory for file dialogs)
            - _model_cache, _model_cache_time (model list cache)

        Worker Mixin reads/writes:
            - _tasks (list of ComparisonTask dicts, protected by _tasks_lock)
            - _task_counter (monotonic task ID counter)
            - _tasks_lock (threading.Lock for _tasks)
            - _tree_item_map (task_id -> treeview iid mapping)
            - _processing (bool, whether any task is running)
            - _start_time, _elapsed_timer_id (elapsed time tracking)
            - _executor (ThreadPoolExecutor for parallel task execution)
            - source_path, output_dir, selected_model, pitch_shift,
              feature_extractor, f0_method, device, output_sample_rate,
              cluster_ratio (read parameters for new tasks)
            - _last_directory (save directory preference)

        Playback Mixin reads/writes:
            - _current_player (subprocess.Popen for current playback)
            - _playback_lock (threading.Lock for player access)
            - volume_var (volume slider value)
            - _tasks (read completed tasks for playback)
            - _tasks_lock (read access to tasks)

        BasePage reads/writes:
            - _cleaned_up (idempotent cleanup flag)
        """
        # ============================================================
        # IMPORTANT: All attributes MUST be initialized BEFORE super().__init__()
        # because BasePage.__init__() calls self._create_widgets() which
        # references these attributes.
        # ============================================================
        import threading

        # ---- Settings manager (fix #1: singleton) ----
        self._settings = SettingsManager()

        # ---- Task management (thread-safe, fix #6) ----
        # Used by: WorkerMixin (add/remove/update tasks), PlaybackMixin (read completed)
        self._tasks = []  # List of ComparisonTask dicts
        self._task_counter = 0
        self._tasks_lock = threading.Lock()  # Protects _tasks list
        self._tree_item_map = {}  # task_id -> treeview iid mapping (fix #4)

        # ---- Model cache (fix #7) ----
        # Used by: UIMixin (model list), WorkerMixin (model validation)
        self._model_cache = []
        self._model_cache_time = 0

        # ---- Processing state ----
        # Used by: WorkerMixin (execution control), UI (button enable/disable)
        self._processing = False
        self._start_time = None
        self._elapsed_timer_id = None

        # ---- Playback state (thread-safe, fix #3) ----
        # Used by: PlaybackMixin (player management), WorkerMixin (stop on cleanup)
        self._current_player = None
        self._playback_lock = threading.Lock()

        # ---- Thread pool (fix #6: configurable max_workers) ----
        max_workers = self._get_max_workers()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # ---- Tkinter variables (UI <-> Worker communication) ----
        # Source/output paths
        self.source_path = tk.StringVar()
        self.output_dir = tk.StringVar()

        # Model and inference parameters
        self.selected_model = tk.StringVar()
        self.pitch_shift = tk.IntVar(value=0)
        self.feature_extractor = tk.StringVar(value="hubert")
        self.f0_method = tk.StringVar(value="rmvpe")
        self.device = tk.StringVar(value="auto")
        self.output_sample_rate = tk.StringVar(value="40000")
        self.cluster_ratio = tk.DoubleVar(value=0.0)

        # Elapsed time display
        self.elapsed_var = tk.StringVar(value="0:00")

        # File info variables
        self.file_info_filename = tk.StringVar(value="未选择文件")
        self.file_info_duration = tk.StringVar(value="--")
        self.file_info_samplerate = tk.StringVar(value="--")
        self.file_info_channels = tk.StringVar(value="--")
        self.file_info_filesize = tk.StringVar(value="--")

        # Volume control
        self.volume_var = tk.DoubleVar(value=0.8)

        # ---- Remembered last directory (from SettingsManager) ----
        self._last_directory = self._settings.get("comparison_last_dir", os.path.expanduser("~"))
        saved_output = self._settings.get("comparison_output_dir", "")
        if saved_output:
            self.output_dir.set(saved_output)

        # ============================================================
        # Now call super().__init__() which triggers _create_widgets()
        # ============================================================
        super().__init__(parent, **kwargs)

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
