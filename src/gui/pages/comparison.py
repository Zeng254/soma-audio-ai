"""
Comparison page for SOMA GUI.

Provides interface for A/B testing voice conversion results with different
parameter configurations. Users can queue multiple comparison tasks, run
them in parallel, and compare the outputs side by side.

Code quality fixes applied:
- SettingsManager singleton for thread-safe settings access
- threading.Lock for task list (fix #6)
- subprocess.Popen for all platforms (fix #3, Windows A/B stop)
- Widget alive guards via safe_after (fix #4)
- Output filename includes task ID (fix #9)
- Model search depth limit (fix #10)
- Shared parameter constants (fix #11)
- Model scan with mtime cache (fix #7)
- Status text constants (fix #5)
- TypedDict for task dicts (fix #3)
- Treeview incremental refresh (fix #4)
- ThreadPoolExecutor cleanup on quit (fix #1)
- Configurable GPU concurrency (fix #6)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import json
import subprocess
import sys
import time
import uuid
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts
from gui.utils import (
    SettingsManager, open_folder, AUDIO_FILETYPES,
    FEATURE_EXTRACTORS, F0_METHODS, DEVICES, SAMPLE_RATES,
    DEFAULT_SAMPLE_RATE, DEFAULT_F0_METHOD, DEFAULT_FEATURE_EXTRACTOR,
    DEFAULT_DEVICE, PITCH_MIN, PITCH_MAX,
    MODEL_CACHE_TTL, MODEL_SEARCH_MAX_DEPTH,
    # Status constants (fix #5)
    STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED,
    STATUS_CANCELLED, STATUS_DISPLAY,
    STATUS_READY, STATUS_ERROR, STATUS_CANCELLED_UI,
    # TypedDict (fix #3)
    ComparisonTask,
    # GPU concurrency (fix #6)
    DEFAULT_MAX_WORKERS_CPU, DEFAULT_MAX_WORKERS_GPU,
    SETTING_KEY_MAX_WORKERS, SETTING_KEY_DEVICE_TYPE,
)


# Column definitions for task list
COLUMNS = ("id", "model", "pitch", "f0", "feature", "status", "time")
COLUMN_HEADINGS = {
    "id": "ID", "model": "Model", "pitch": "Pitch",
    "f0": "F0", "feature": "Feature", "status": "Status", "time": "Time",
}
COLUMN_WIDTHS = {
    "id": 40, "model": 120, "pitch": 50,
    "f0": 60, "feature": 70, "status": 90, "time": 60,
}


class ComparisonPage(BasePage):
    """
    Comparison page for A/B testing voice conversion results.

    Features:
    - Multi-task queue with independent parameter sets
    - Thread pool for parallel processing (configurable max_workers)
    - Per-task config: model, pitch, F0 method, feature extractor,
      cluster ratio, device, sample rate
    - Task list with status tracking
    - Batch start, cancel all, clear done
    - Duplicate last task config for quick setup
    - A/B audio switching
    - Playback via system default player (subprocess.Popen, cross-platform)
    - Volume control
    - Batch export with auto-naming
    - Config save/load as JSON
    """

    PAGE_NAME = "Compare"
    PAGE_ICON = "\U0001f504"
    PAGE_DESCRIPTION = "A/B compare results"

    # Use shared constants (fix #11)
    FEATURE_EXTRACTORS = FEATURE_EXTRACTORS
    F0_METHODS = F0_METHODS
    DEVICES = DEVICES
    SAMPLE_RATES = SAMPLE_RATES

    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the comparison page."""
        super().__init__(parent, app)

        # Settings manager (singleton, thread-safe)
        self._settings = SettingsManager()

        # Task list with lock (fix #6)
        self._tasks: List[ComparisonTask] = []
        self._tasks_lock = threading.Lock()
        self._task_counter = 0

        # Treeview item mapping for incremental refresh (fix #4)
        self._tree_item_map: Dict[int, str] = {}  # task_id -> treeview iid

        # Thread pool for parallel task execution (fix #6: configurable)
        max_workers = self._get_max_workers()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Playback state (cross-platform, fix #3)
        self._current_player: Optional[subprocess.Popen] = None
        self._playback_lock = threading.Lock()

        # State
        self._processing = False
        self._start_time: Optional[float] = None
        self._elapsed_timer_id: Optional[str] = None

        # Model cache (fix #7)
        self._model_cache: List[str] = []
        self._model_cache_time: float = 0.0

        # Variables
        self.source_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.pitch_shift = tk.IntVar(value=0)
        self.feature_extractor = tk.StringVar(value=DEFAULT_FEATURE_EXTRACTOR)
        self.f0_method = tk.StringVar(value=DEFAULT_F0_METHOD)
        self.device = tk.StringVar(value=DEFAULT_DEVICE)
        self.output_sample_rate = tk.StringVar(value=DEFAULT_SAMPLE_RATE)
        self.cluster_ratio = tk.DoubleVar(value=0.0)
        self.elapsed_var = tk.StringVar(value="0:00")

        # File info variables
        self.file_info_duration = tk.StringVar(value="--")
        self.file_info_samplerate = tk.StringVar(value="--")
        self.file_info_channels = tk.StringVar(value="--")
        self.file_info_filesize = tk.StringVar(value="--")
        self.file_info_filename = tk.StringVar(value="No file selected")

        # Remembered last directory
        self._last_directory = self._settings.get(
            "comparison_last_dir", os.path.expanduser("~")
        )

    def _get_max_workers(self) -> int:
        """Get configured max workers for thread pool (fix #6)."""
        try:
            stored = self._settings.get(SETTING_KEY_MAX_WORKERS, None)
            if stored is not None:
                return max(1, int(stored))
        except (ValueError, TypeError):
            pass
        # Auto-detect based on device
        device_type = self._settings.get(SETTING_KEY_DEVICE_TYPE, "auto")
        if device_type == "cuda":
            return DEFAULT_MAX_WORKERS_GPU
        return DEFAULT_MAX_WORKERS_CPU

    def cleanup(self):
        """Shut down thread pool on app exit (fix #1)."""
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
        self._stop_playback()

    def _create_widgets(self):
        """Create comparison page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Effect Comparison",
            "Compare voice conversion results with different parameters"
        )

        # Main content: left (config) + right (tasks + results)
        main_frame = ttk.Frame(self.content_frame, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left column - Configuration
        left_frame = ttk.Frame(main_frame, style="TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left_frame.configure(width=320)

        self._create_dropzone_section(left_frame)
        self._create_file_info_section(left_frame)
        self._create_task_config_section(left_frame)
        self._create_action_section(left_frame)

        # Right column - Tasks and Results
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._create_task_list_section(right_frame)
        self._create_playback_section(right_frame)
        self._create_result_log_section(right_frame)
        self._create_export_section(right_frame)

    # ── Drop Zone ──────────────────────────────────────────────────────

    def _create_dropzone_section(self, parent: tk.Widget):
        """Create a visual drop zone for source audio files."""
        card = self.create_card(parent, "Source Audio")

        # Drop zone visual
        self.dropzone = tk.Frame(
            card, bg=Colors.BG_INPUT,
            highlightbackground=Colors.BORDER, highlightthickness=2,
            padx=20, pady=10,
        )
        self.dropzone.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            self.dropzone, text="\U0001f3a4",
            font=(Fonts.FAMILY, 24), bg=Colors.BG_INPUT, fg=Colors.TEXT_MUTED,
        ).pack(pady=(5, 0))

        tk.Label(
            self.dropzone,
            text="Click 'Browse' or drag audio file here",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            bg=Colors.BG_INPUT, fg=Colors.TEXT_SECONDARY,
        ).pack(pady=(0, 5))

        # Path entry with browse button
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        path_entry = ttk.Entry(path_frame, textvariable=self.source_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(
            path_frame, text="Browse...",
            style="Secondary.TButton", command=self._browse_source
        )
        browse_btn.pack(side=tk.RIGHT)

        # Bind trace to update file info
        self.source_path.trace_add("write", self._on_source_path_changed)

    # ── File Info ──────────────────────────────────────────────────────

    def _create_file_info_section(self, parent: tk.Widget):
        """Create file information display section."""
        card = self.create_card(parent, "File Info")

        info_frame = ttk.Frame(card, style="Card.TFrame")
        info_frame.pack(fill=tk.X)

        info_items = [
            ("File", self.file_info_filename),
            ("Duration", self.file_info_duration),
            ("Sample Rate", self.file_info_samplerate),
            ("Channels", self.file_info_channels),
            ("File Size", self.file_info_filesize),
        ]

        for label_text, var in info_items:
            row = ttk.Frame(info_frame, style="Card.TFrame")
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label_text}:", style="Card.TLabel", width=12).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=var, style="Muted.TLabel").pack(side=tk.LEFT, padx=(5, 0))

    # ── Task Configuration ─────────────────────────────────────────────

    def _create_task_config_section(self, parent: tk.Widget):
        """Create task parameter configuration section."""
        card = self.create_card(parent, "Task Configuration")

        # Model selection
        model_frame = ttk.Frame(card, style="Card.TFrame")
        model_frame.pack(fill=tk.X, pady=3)
        ttk.Label(model_frame, text="Model:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        model_combo = ttk.Combobox(
            model_frame, textvariable=self.selected_model, state="readonly"
        )
        model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        refresh_btn = ttk.Button(
            model_frame, text="\U0001f504",
            style="Secondary.TButton", command=self._refresh_models
        )
        refresh_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Pitch shift
        pitch_frame = ttk.Frame(card, style="Card.TFrame")
        pitch_frame.pack(fill=tk.X, pady=3)
        ttk.Label(pitch_frame, text="Pitch:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        pitch_spinbox = tk.Spinbox(
            pitch_frame, from_=PITCH_MIN, to=PITCH_MAX,
            textvariable=self.pitch_shift,
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            buttonbackground=Colors.BG_TERTIARY, width=5
        )
        pitch_spinbox.pack(side=tk.LEFT)
        ttk.Label(pitch_frame, text="semitones", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Feature extractor
        fe_frame = ttk.Frame(card, style="Card.TFrame")
        fe_frame.pack(fill=tk.X, pady=3)
        ttk.Label(fe_frame, text="Feature:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        fe_combo = ttk.Combobox(
            fe_frame, textvariable=self.feature_extractor,
            values=list(self.FEATURE_EXTRACTORS.keys()), state="readonly", width=12
        )
        fe_combo.pack(side=tk.LEFT)

        # F0 method
        f0_frame = ttk.Frame(card, style="Card.TFrame")
        f0_frame.pack(fill=tk.X, pady=3)
        ttk.Label(f0_frame, text="F0 Method:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        f0_combo = ttk.Combobox(
            f0_frame, textvariable=self.f0_method,
            values=list(self.F0_METHODS.keys()), state="readonly", width=12
        )
        f0_combo.pack(side=tk.LEFT)

        # Device
        dev_frame = ttk.Frame(card, style="Card.TFrame")
        dev_frame.pack(fill=tk.X, pady=3)
        ttk.Label(dev_frame, text="Device:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        dev_combo = ttk.Combobox(
            dev_frame, textvariable=self.device,
            values=list(self.DEVICES.keys()), state="readonly", width=12
        )
        dev_combo.pack(side=tk.LEFT)

        # Sample rate
        sr_frame = ttk.Frame(card, style="Card.TFrame")
        sr_frame.pack(fill=tk.X, pady=3)
        ttk.Label(sr_frame, text="Sample Rate:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        sr_combo = ttk.Combobox(
            sr_frame, textvariable=self.output_sample_rate,
            values=self.SAMPLE_RATES, state="readonly", width=12
        )
        sr_combo.pack(side=tk.LEFT)

        # Cluster ratio
        cluster_frame = ttk.Frame(card, style="Card.TFrame")
        cluster_frame.pack(fill=tk.X, pady=3)
        ttk.Label(cluster_frame, text="Cluster:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        cluster_scale = ttk.Scale(
            cluster_frame, from_=0.0, to=1.0,
            variable=self.cluster_ratio, orient=tk.HORIZONTAL
        )
        cluster_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.cluster_label = ttk.Label(cluster_frame, text="0.00", style="Card.TLabel", width=5)
        self.cluster_label.pack(side=tk.LEFT)
        self.cluster_ratio.trace_add("write", self._update_cluster_label)

        # Output directory
        outdir_frame = ttk.Frame(card, style="Card.TFrame")
        outdir_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(outdir_frame, text="Output Dir:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        outdir_entry = ttk.Entry(outdir_frame, textvariable=self.output_dir)
        outdir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        outdir_btn = ttk.Button(
            outdir_frame, text="...",
            style="Secondary.TButton", command=self._browse_output_dir
        )
        outdir_btn.pack(side=tk.RIGHT)

        self._refresh_models()

    def _update_cluster_label(self, *args):
        """Update cluster ratio display label."""
        val = self.cluster_ratio.get()
        self.cluster_label.configure(text=f"{val:.2f}")

    # ── Action Buttons ─────────────────────────────────────────────────

    def _create_action_section(self, parent: tk.Widget):
        """Create action buttons section."""
        card = self.create_card(parent, "Actions")

        # Add task button
        add_btn = ttk.Button(
            card, text="\u2795 Add Task",
            style="Primary.TButton", command=self._add_task
        )
        add_btn.pack(fill=tk.X, pady=(0, 5))

        # Duplicate last task
        dup_btn = ttk.Button(
            card, text="\U0001f4cb Duplicate Last Config",
            style="Secondary.TButton", command=self._duplicate_last_task
        )
        dup_btn.pack(fill=tk.X, pady=(0, 5))

        # Batch operations
        batch_frame = ttk.Frame(card, style="Card.TFrame")
        batch_frame.pack(fill=tk.X, pady=(5, 0))

        start_all_btn = ttk.Button(
            batch_frame, text="\u25b6 Start All",
            style="Primary.TButton", command=self._start_all_tasks
        )
        start_all_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        cancel_all_btn = ttk.Button(
            batch_frame, text="\u23f9 Cancel All",
            style="Danger.TButton", command=self._cancel_all_tasks
        )
        cancel_all_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        # Clear / Remove
        clear_frame = ttk.Frame(card, style="Card.TFrame")
        clear_frame.pack(fill=tk.X, pady=(5, 0))

        clear_done_btn = ttk.Button(
            clear_frame, text="Clear Done",
            style="Secondary.TButton", command=self._clear_done_tasks
        )
        clear_done_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        remove_btn = ttk.Button(
            clear_frame, text="Remove Selected",
            style="Secondary.TButton", command=self._remove_selected_task
        )
        remove_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        # Config save/load
        config_frame = ttk.Frame(card, style="Card.TFrame")
        config_frame.pack(fill=tk.X, pady=(10, 0))

        save_cfg_btn = ttk.Button(
            config_frame, text="Save Config",
            style="Secondary.TButton", command=self._save_config
        )
        save_cfg_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        load_cfg_btn = ttk.Button(
            config_frame, text="Load Config",
            style="Secondary.TButton", command=self._load_config
        )
        load_cfg_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

    # ── Task List ──────────────────────────────────────────────────────

    def _create_task_list_section(self, parent: tk.Widget):
        """Create task list section with Treeview."""
        card = self.create_card(parent, "Task Queue")

        # Treeview for task list
        tree_frame = ttk.Frame(card, style="Card.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.task_tree = ttk.Treeview(
            tree_frame, columns=COLUMNS,
            show="headings", height=8
        )

        for col in COLUMNS:
            self.task_tree.heading(col, text=COLUMN_HEADINGS[col])
            self.task_tree.column(col, width=COLUMN_WIDTHS[col])

        scrollbar = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self.task_tree.yview
        )
        self.task_tree.configure(yscrollcommand=scrollbar.set)

        self.task_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Task count label
        self.task_count_label = ttk.Label(
            card, text="0 tasks", style="Muted.TLabel"
        )
        self.task_count_label.pack(anchor=tk.W, pady=(5, 0))

    # ── Playback Section ───────────────────────────────────────────────

    def _create_playback_section(self, parent: tk.Widget):
        """Create A/B playback controls section."""
        card = self.create_card(parent, "Playback (A/B Compare)")

        # A/B selection info
        ab_frame = ttk.Frame(card, style="Card.TFrame")
        ab_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ab_frame, text="Select 1-2 completed tasks for A/B comparison", style="Muted.TLabel").pack(
            anchor=tk.W
        )

        # Playback buttons
        play_frame = ttk.Frame(card, style="Card.TFrame")
        play_frame.pack(fill=tk.X, pady=(0, 10))

        play_a_btn = ttk.Button(
            play_frame, text="\u25b6 Play A",
            style="Primary.TButton", command=self._play_selected_a
        )
        play_a_btn.pack(side=tk.LEFT, padx=(0, 5))

        play_b_btn = ttk.Button(
            play_frame, text="\u25b6 Play B",
            style="Primary.TButton", command=self._play_selected_b
        )
        play_b_btn.pack(side=tk.LEFT, padx=(0, 5))

        stop_btn = ttk.Button(
            play_frame, text="\u23f9 Stop",
            style="Danger.TButton", command=self._stop_playback
        )
        stop_btn.pack(side=tk.LEFT, padx=(0, 5))

        ab_btn = ttk.Button(
            play_frame, text="\U0001f504 A/B Switch",
            style="Secondary.TButton", command=self._ab_switch_play
        )
        ab_btn.pack(side=tk.LEFT)

        # Volume control
        vol_frame = ttk.Frame(card, style="Card.TFrame")
        vol_frame.pack(fill=tk.X)
        ttk.Label(vol_frame, text="Volume:", style="Card.TLabel").pack(side=tk.LEFT)
        self.volume_var = tk.DoubleVar(value=0.8)
        vol_scale = ttk.Scale(
            vol_frame, from_=0.0, to=1.0,
            variable=self.volume_var, orient=tk.HORIZONTAL
        )
        vol_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

    # ── Result Log ─────────────────────────────────────────────────────

    def _create_result_log_section(self, parent: tk.Widget):
        """Create result log section."""
        card = self.create_card(parent, "Result Log")

        log_frame = ttk.Frame(card, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_display = tk.Text(
            log_frame, height=8,
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            wrap=tk.WORD, state=tk.DISABLED
        )
        self.log_display.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.log_display, command=self.log_display.yview)
        self.log_display.configure(yscrollcommand=scrollbar.set)

    # ── Export Section ─────────────────────────────────────────────────

    def _create_export_section(self, parent: tk.Widget):
        """Create export section."""
        card = self.create_card(parent, "Export Results")

        # Elapsed time
        time_frame = ttk.Frame(card, style="Card.TFrame")
        time_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(time_frame, text="Elapsed:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(time_frame, textvariable=self.elapsed_var, style="Card.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Export buttons
        export_frame = ttk.Frame(card, style="Card.TFrame")
        export_frame.pack(fill=tk.X)

        export_all_btn = ttk.Button(
            export_frame, text="\U0001f4e6 Export All Results",
            style="Primary.TButton", command=self._export_all_results
        )
        export_all_btn.pack(side=tk.LEFT, padx=(0, 10))

        open_folder_btn = ttk.Button(
            export_frame, text="\U0001f4c1 Open Folder",
            style="Secondary.TButton", command=self._open_output_folder
        )
        open_folder_btn.pack(side=tk.LEFT)

    # ── Browse Handlers ────────────────────────────────────────────────

    def _browse_source(self):
        """Browse for source audio file."""
        filename = filedialog.askopenfilename(
            title="Select Source Audio",
            filetypes=AUDIO_FILETYPES,
            initialdir=self._last_directory,
        )
        if filename:
            self.source_path.set(filename)
            self._last_directory = os.path.dirname(filename)
            self._settings.set("comparison_last_dir", self._last_directory)

            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(filename))

    def _browse_output_dir(self):
        """Browse for output directory."""
        dirname = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self._last_directory,
        )
        if dirname:
            self.output_dir.set(dirname)
            self._last_directory = dirname
            self._settings.set("comparison_last_dir", dirname)

    # ── File Info (background thread, fix #12) ─────────────────────────

    def _on_source_path_changed(self, *args):
        """Triggered when source_path changes."""
        filepath = self.source_path.get()
        if not filepath or not os.path.isfile(filepath):
            self._reset_file_info()
            return
        threading.Thread(target=self._load_file_info, args=(filepath,), daemon=True).start()

    def _reset_file_info(self):
        """Reset file info to defaults."""
        self.file_info_filename.set("No file selected")
        self.file_info_duration.set("--")
        self.file_info_samplerate.set("--")
        self.file_info_channels.set("--")
        self.file_info_filesize.set("--")

    def _load_file_info(self, filepath: str):
        """Load audio file info in background thread."""
        try:
            size_bytes = os.path.getsize(filepath)
            size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / 1024 / 1024:.2f} MB"
            filename = os.path.basename(filepath)
            self.safe_after(0, lambda: self.file_info_filename.set(filename))
            self.safe_after(0, lambda: self.file_info_filesize.set(size_str))

            try:
                import soundfile as sf
                info = sf.info(filepath)
                duration_sec = info.duration
                minutes = int(duration_sec // 60)
                seconds = int(duration_sec % 60)
                self.safe_after(0, lambda: self.file_info_duration.set(f"{minutes}:{seconds:02d}"))
                self.safe_after(0, lambda: self.file_info_samplerate.set(f"{info.samplerate} Hz"))
                ch_map = {1: "Mono", 2: "Stereo"}
                self.safe_after(0, lambda: self.file_info_channels.set(
                    ch_map.get(info.channels, f"{info.channels} ch")
                ))
            except Exception:
                self.safe_after(0, lambda: self.file_info_duration.set("N/A"))
                self.safe_after(0, lambda: self.file_info_samplerate.set("N/A"))
                self.safe_after(0, lambda: self.file_info_channels.set("N/A"))
        except Exception:
            self.safe_after(0, self._reset_file_info)

    # ── Model Management (with cache, fix #7, depth limit, fix #10) ────

    def _refresh_models(self):
        """Refresh available models with mtime-based cache."""
        now = time.time()
        if self._model_cache and (now - self._model_cache_time) < MODEL_CACHE_TTL:
            models = self._model_cache
        else:
            model_dirs = [
                os.path.join(os.path.expanduser("~"), ".soma", "models"),
                os.path.join(os.getcwd(), "assets", "models"),
            ]
            models = []
            for model_dir in model_dirs:
                if os.path.isdir(model_dir):
                    for f in os.listdir(model_dir):
                        if f.endswith(".pth"):
                            models.append(f[:-4])
            self._model_cache = models
            self._model_cache_time = now

        if not models:
            models = ["No models available"]

        # Update all model comboboxes in the config section
        for widget in self._find_comboboxes():
            widget.configure(values=models)
            if not widget.get():
                widget.set(models[0])

    def _find_comboboxes(self) -> List[ttk.Combobox]:
        """Find all model comboboxes in the config section."""
        result = []
        for widget in self.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ttk.Combobox):
                    cv = getattr(child, '_textvariable', None)
                    if cv == self.selected_model:
                        result.append(child)
        return result

    def _find_model_file(self, model_name: str) -> Optional[str]:
        """Find model file by name (depth-limited, fix #10)."""
        search_dirs = [
            os.path.join(os.path.expanduser("~"), ".soma", "models"),
            os.path.join(os.getcwd(), "assets", "models"),
        ]
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            direct = os.path.join(search_dir, f"{model_name}.pth")
            if os.path.isfile(direct):
                return direct
            for root, dirs, files in os.walk(search_dir):
                rel = os.path.relpath(root, search_dir)
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if depth >= MODEL_SEARCH_MAX_DEPTH:
                    dirs.clear()
                    continue
                for f in files:
                    if f == f"{model_name}.pth":
                        return os.path.join(root, f)
        return None

    # ── Task Management (thread-safe, fix #6) ──────────────────────────

    def _get_current_config(self) -> Dict:
        """Get current parameter configuration."""
        return {
            "model": self.selected_model.get(),
            "pitch": self.pitch_shift.get(),
            "feature_extractor": self.feature_extractor.get(),
            "f0_method": self.f0_method.get(),
            "device": self.device.get(),
            "sample_rate": self.output_sample_rate.get(),
            "cluster_ratio": round(self.cluster_ratio.get(), 2),
        }

    def _add_task(self):
        """Add a new task with current configuration."""
        if not self.source_path.get():
            messagebox.showwarning("Warning", "Please select a source audio file first.")
            return

        config = self._get_current_config()
        if config["model"] == "No models available":
            messagebox.showwarning("Warning", "No voice models available.")
            return

        with self._tasks_lock:
            self._task_counter += 1
            task_id = self._task_counter
            task: ComparisonTask = {
                "id": task_id,
                "config": config,
                "status": STATUS_QUEUED,
                "result_path": None,
                "error": None,
                "duration": None,
                "cancel_flag": threading.Event(),
                "uuid": uuid.uuid4().hex[:8],  # fix #9: unique ID for filenames
            }
            self._tasks.append(task)

        # Insert into treeview and track in item map (fix #4)
        iid = str(task_id)
        self.task_tree.insert("", tk.END, iid=iid, values=(
            task_id,
            config["model"],
            f"{config['pitch']:+d}",
            config["f0_method"],
            config["feature_extractor"],
            STATUS_DISPLAY[STATUS_QUEUED],
            "--",
        ))
        self._tree_item_map[task_id] = iid

        self._update_task_count()
        self._log(f"Task #{task_id} added: model={config['model']}, pitch={config['pitch']:+d}, "
                  f"f0={config['f0_method']}, feature={config['feature_extractor']}")

    def _duplicate_last_task(self):
        """Duplicate the last task's configuration for quick setup."""
        with self._tasks_lock:
            if not self._tasks:
                messagebox.showinfo("Info", "No tasks to duplicate.")
                return
            last_config = self._tasks[-1]["config"].copy()

        # Apply last config to current controls
        self.selected_model.set(last_config["model"])
        self.pitch_shift.set(last_config["pitch"])
        self.feature_extractor.set(last_config["feature_extractor"])
        self.f0_method.set(last_config["f0_method"])
        self.device.set(last_config["device"])
        self.output_sample_rate.set(last_config["sample_rate"])
        self.cluster_ratio.set(last_config["cluster_ratio"])

        self._log(f"Duplicated config from last task: model={last_config['model']}")

    def _remove_selected_task(self):
        """Remove selected task from queue."""
        selection = self.task_tree.selection()
        if not selection:
            return

        for item_id in selection:
            task_id = int(item_id)
            with self._tasks_lock:
                task = next((t for t in self._tasks if t["id"] == task_id), None)
                if task:
                    if task["status"] == STATUS_RUNNING:
                        task["cancel_flag"].set()
                    self._tasks.remove(task)
            try:
                self.task_tree.delete(item_id)
            except tk.TclError:
                pass
            # Clean up item map (fix #4)
            self._tree_item_map.pop(task_id, None)

        self._update_task_count()

    def _clear_done_tasks(self):
        """Clear all completed/failed/cancelled tasks."""
        with self._tasks_lock:
            done_tasks = [t for t in self._tasks if t["status"] in (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)]
            for task in done_tasks:
                self._tasks.remove(task)
                task_id = task["id"]
                try:
                    self.task_tree.delete(str(task_id))
                except tk.TclError:
                    pass
                # Clean up item map (fix #4)
                self._tree_item_map.pop(task_id, None)

        self._update_task_count()

    def _update_task_count(self):
        """Update task count label."""
        with self._tasks_lock:
            total = len(self._tasks)
            done = sum(1 for t in self._tasks if t["status"] == STATUS_DONE)
            running = sum(1 for t in self._tasks if t["status"] == STATUS_RUNNING)
        self.task_count_label.configure(text=f"{total} tasks ({done} done, {running} running)")

    def _update_task_in_tree(self, task_id: int, task: ComparisonTask):
        """Update a task's display in the treeview (main thread, incremental fix #4).

        Uses the item mapping to update only the changed row instead of
        deleting and re-inserting all rows.
        """
        try:
            iid = str(task_id)
            values = (
                task_id,
                task["config"]["model"],
                f"{task['config']['pitch']:+d}",
                task["config"]["f0_method"],
                task["config"]["feature_extractor"],
                STATUS_DISPLAY.get(task["status"], task["status"]),
                f"{task['duration']:.1f}s" if task["duration"] else "--",
            )
            if iid in self._tree_item_map:
                # Incremental update: just update the existing row
                self.task_tree.item(iid, values=values)
            else:
                # Row doesn't exist yet, insert it
                new_iid = self.task_tree.insert("", tk.END, iid=iid, values=values)
                self._tree_item_map[task_id] = new_iid
        except tk.TclError:
            pass

    # ── Batch Operations ───────────────────────────────────────────────

    def _start_all_tasks(self):
        """Start all queued tasks."""
        if not self.source_path.get():
            messagebox.showwarning("Warning", "Please select a source audio file first.")
            return

        if not self.output_dir.get():
            messagebox.showwarning("Warning", "Please specify an output directory first.")
            return

        try:
            os.makedirs(self.output_dir.get(), exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error", f"Cannot create output directory:\n{e}")
            return

        with self._tasks_lock:
            queued = [t for t in self._tasks if t["status"] == STATUS_QUEUED]

        if not queued:
            messagebox.showinfo("Info", "No queued tasks to start.")
            return

        self._processing = True
        self._start_time = time.time()
        self._tick_elapsed()

        for task in queued:
            task["cancel_flag"].clear()
            task["status"] = STATUS_RUNNING
            self.safe_after(0, lambda t=task: self._update_task_in_tree(t["id"], t))
            self._executor.submit(self._run_task, task)

        self._update_task_count()
        self._log(f"Started {len(queued)} task(s)")

    def _cancel_all_tasks(self):
        """Cancel all running tasks."""
        with self._tasks_lock:
            running = [t for t in self._tasks if t["status"] == STATUS_RUNNING]

        for task in running:
            task["cancel_flag"].set()

        self._log(f"Cancelled {len(running)} task(s)")

    # ── Task Execution ─────────────────────────────────────────────────

    def _run_task(self, task: Dict):
        """Execute a single conversion task (runs in thread pool)."""
        task_id = task["id"]
        config = task["config"]
        cancel_flag = task["cancel_flag"]
        start_time = time.time()

        try:
            import numpy as np
            import soundfile as sf
            from training.inference import RVCInference

            # Find model file
            model_path = self._find_model_file(config["model"])
            if model_path is None:
                raise FileNotFoundError(f"Model '{config['model']}' not found")

            if cancel_flag.is_set():
                raise InterruptedError("Cancelled")

            # Determine device
            device_str = config["device"]
            if device_str == "auto":
                try:
                    import torch
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device_str = "cpu"

            output_sr = int(config["sample_rate"])

            pipeline = RVCInference(
                model_path=model_path,
                device=device_str,
                output_sample_rate=output_sr,
                f0_method=config["f0_method"],
            )

            if cancel_flag.is_set():
                raise InterruptedError("Cancelled")

            # Load audio
            audio, sr = sf.read(self.source_path.get())
            if audio.ndim == 2 and audio.shape[1] > 2:
                audio = np.mean(audio, axis=1)

            if cancel_flag.is_set():
                raise InterruptedError("Cancelled")

            # Convert
            transpose = float(config["pitch"])
            result = pipeline.convert(audio, sample_rate=sr, transpose=transpose)

            if cancel_flag.is_set():
                raise InterruptedError("Cancelled")

            # Save with task ID in filename (fix #9)
            source_name = os.path.splitext(os.path.basename(self.source_path.get()))[0]
            output_filename = (
                f"{source_name}_t{task_id}_{config['model']}_"
                f"pitch{config['pitch']:+d}_{config['f0_method']}_"
                f"{config['feature_extractor']}.wav"
            )
            output_path = os.path.join(self.output_dir.get(), output_filename)
            sf.write(output_path, result, output_sr)

            task["status"] = STATUS_DONE
            task["result_path"] = output_path
            task["duration"] = time.time() - start_time

            self.safe_after(0, lambda: self._log(
                f"Task #{task_id} completed in {task['duration']:.1f}s"
            ))

        except InterruptedError:
            task["status"] = STATUS_CANCELLED
            task["duration"] = time.time() - start_time
            self.safe_after(0, lambda: self._log(f"Task #{task_id} cancelled"))

        except Exception as e:
            task["status"] = STATUS_FAILED
            task["error"] = str(e)
            task["duration"] = time.time() - start_time
            self.safe_after(0, lambda: self._log(f"Task #{task_id} failed: {e}"))

        finally:
            # Update treeview and count from main thread
            self.safe_after(0, lambda: self._update_task_in_tree(task_id, task))
            self.safe_after(0, self._update_task_count)
            self.safe_after(0, self._check_all_done)

    def _check_all_done(self):
        """Check if all tasks are done and stop the timer."""
        with self._tasks_lock:
            all_done = all(
                t["status"] in (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)
                for t in self._tasks
            )
        if all_done and self._processing:
            self._processing = False
            if self._start_time:
                elapsed = time.time() - self._start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                self.elapsed_var.set(f"{minutes}:{seconds:02d} (all done)")

            with self._tasks_lock:
                done_count = sum(1 for t in self._tasks if t["status"] == STATUS_DONE)
            self._log(f"All tasks finished. {done_count} succeeded.")

            if done_count > 0:
                result = messagebox.askyesno(
                    "Comparison Complete",
                    f"All comparison tasks finished!\n\n"
                    f"{done_count} result(s) saved to:\n{self.output_dir.get()}\n\n"
                    f"Open output folder?"
                )
                if result:
                    self._open_output_folder()

    # ── Playback (cross-platform, fix #3) ──────────────────────────────

    def _play_audio_file(self, filepath: str):
        """Play audio file using system default player (cross-platform, fix #3).

        Uses subprocess.Popen on all platforms for consistent stop behavior.
        On macOS: afplay (can be killed)
        On Linux: xdg-open
        On Windows: powershell -c (Start-Player) or ffplay if available
        """
        if not os.path.isfile(filepath):
            messagebox.showwarning("Warning", f"File not found:\n{filepath}")
            return

        self._stop_playback()

        try:
            if sys.platform == "darwin":
                # macOS: afplay can be killed via process group
                proc = subprocess.Popen(
                    ["afplay", filepath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif sys.platform == "win32":
                # Windows: use PowerShell to play via Windows Media Player COM
                # This allows us to get a process handle and kill it
                ps_cmd = (
                    f'$player = New-Object System.Media.SoundPlayer("{filepath}"); '
                    f'$player.Play(); '
                    f'Start-Sleep -Seconds 999'
                )
                proc = subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
                )
            else:
                # Linux: xdg-open
                proc = subprocess.Popen(
                    ["xdg-open", filepath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            with self._playback_lock:
                self._current_player = proc

        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio:\n{e}")

    def _stop_playback(self):
        """Stop current playback (cross-platform, fix #3)."""
        with self._playback_lock:
            if self._current_player is not None:
                try:
                    self._current_player.terminate()
                    self._current_player.wait(timeout=2)
                except Exception:
                    try:
                        self._current_player.kill()
                    except Exception:
                        pass
                self._current_player = None

    def _play_selected_a(self):
        """Play the first selected completed task."""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Select a completed task to play.")
            return

        task_id = int(selection[0])
        with self._tasks_lock:
            task = next((t for t in self._tasks if t["id"] == task_id), None)

        if not task or task["status"] != STATUS_DONE or not task["result_path"]:
            messagebox.showwarning("Warning", "Selected task has no completed result.")
            return

        self._play_audio_file(task["result_path"])
        self._log(f"Playing A: Task #{task_id}")

    def _play_selected_b(self):
        """Play the second selected completed task (or first if only one selected)."""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Select a completed task to play.")
            return

        # Use second selected item if available, otherwise first
        idx = min(1, len(selection) - 1)
        task_id = int(selection[idx])
        with self._tasks_lock:
            task = next((t for t in self._tasks if t["id"] == task_id), None)

        if not task or task["status"] != STATUS_DONE or not task["result_path"]:
            messagebox.showwarning("Warning", "Selected task has no completed result.")
            return

        self._play_audio_file(task["result_path"])
        self._log(f"Playing B: Task #{task_id}")

    def _ab_switch_play(self):
        """A/B switch playback: play first selected, then second after a delay."""
        selection = self.task_tree.selection()
        if len(selection) < 2:
            messagebox.showinfo("Info", "Select exactly 2 completed tasks for A/B comparison.")
            return

        task_a_id = int(selection[0])
        task_b_id = int(selection[1])

        with self._tasks_lock:
            task_a = next((t for t in self._tasks if t["id"] == task_a_id), None)
            task_b = next((t for t in self._tasks if t["id"] == task_b_id), None)

        if not task_a or not task_b:
            messagebox.showwarning("Warning", "Could not find selected tasks.")
            return

        if task_a["status"] != STATUS_DONE or task_b["status"] != STATUS_DONE:
            messagebox.showwarning("Warning", "Both tasks must be completed.")
            return

        if not task_a["result_path"] or not task_b["result_path"]:
            messagebox.showwarning("Warning", "Result files not found.")
            return

        # Play A first, then B after a delay
        self._play_audio_file(task_a["result_path"])
        self._log(f"A/B Switch: Playing A (Task #{task_a_id})...")

        # Schedule B playback after 5 seconds
        self.safe_after(5000, lambda: self._play_b_after_a(task_b_id, task_b["result_path"]))

    def _play_b_after_a(self, task_id: int, filepath: str):
        """Play task B after A has played (scheduled callback)."""
        self._stop_playback()
        self._play_audio_file(filepath)
        self._log(f"A/B Switch: Playing B (Task #{task_id})...")

    # ── Export ──────────────────────────────────────────────────────────

    def _export_all_results(self):
        """Export all completed results to a chosen directory."""
        with self._tasks_lock:
            done_tasks = [t for t in self._tasks if t["status"] == STATUS_DONE and t["result_path"]]

        if not done_tasks:
            messagebox.showinfo("Info", "No completed results to export.")
            return

        target_dir = filedialog.askdirectory(
            title="Select Export Directory",
            initialdir=self._last_directory,
        )
        if not target_dir:
            return

        import shutil
        exported = 0
        for task in done_tasks:
            source_path = task["result_path"]
            if source_path and os.path.isfile(source_path):
                filename = os.path.basename(source_path)
                dest_path = os.path.join(target_dir, filename)
                try:
                    shutil.copy2(source_path, dest_path)
                    exported += 1
                except Exception as e:
                    self._log(f"Export failed for {filename}: {e}")

        self._log(f"Exported {exported} file(s) to {target_dir}")
        messagebox.showinfo("Export Complete", f"Exported {exported} file(s) to:\n{target_dir}")

    def _open_output_folder(self):
        """Open output folder using common utility."""
        if self.output_dir.get() and os.path.exists(self.output_dir.get()):
            if not open_folder(self.output_dir.get()):
                messagebox.showwarning("Warning", "Could not open folder.")
        else:
            messagebox.showwarning("Warning", "Output folder does not exist.")

    # ── Config Save/Load ───────────────────────────────────────────────

    def _save_config(self):
        """Save current task configurations to JSON."""
        with self._tasks_lock:
            configs = [t["config"].copy() for t in self._tasks]

        if not configs:
            messagebox.showinfo("Info", "No tasks to save.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Comparison Config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self._last_directory,
        )
        if not filepath:
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"tasks": configs, "source": self.source_path.get()}, f, indent=2)
            self._log(f"Config saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{e}")

    def _load_config(self):
        """Load task configurations from JSON."""
        filepath = filedialog.askopenfilename(
            title="Load Comparison Config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self._last_directory,
        )
        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            configs = data.get("tasks", [])
            source = data.get("source", "")

            if source and os.path.isfile(source):
                self.source_path.set(source)

            for config in configs:
                with self._tasks_lock:
                    self._task_counter += 1
                    task_id = self._task_counter
                    task: ComparisonTask = {
                        "id": task_id,
                        "config": config,
                        "status": STATUS_QUEUED,
                        "result_path": None,
                        "error": None,
                        "duration": None,
                        "cancel_flag": threading.Event(),
                        "uuid": uuid.uuid4().hex[:8],
                    }
                    self._tasks.append(task)

                iid = str(task_id)
                self.task_tree.insert("", tk.END, iid=iid, values=(
                    task_id,
                    config.get("model", "?"),
                    f"{config.get('pitch', 0):+d}",
                    config.get("f0_method", "?"),
                    config.get("feature_extractor", "?"),
                    STATUS_DISPLAY[STATUS_QUEUED],
                    "--",
                ))
                self._tree_item_map[task_id] = iid

            self._update_task_count()
            self._log(f"Loaded {len(configs)} task(s) from {filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    # ── Logging ────────────────────────────────────────────────────────

    def _log(self, message: str):
        """Add message to log display (main thread only)."""
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.log_display.configure(state=tk.DISABLED)

    # ── Elapsed Timer ──────────────────────────────────────────────────

    def _tick_elapsed(self):
        """Update elapsed time display."""
        if self._start_time is not None and self._processing:
            elapsed = time.time() - self._start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.elapsed_var.set(f"{minutes}:{seconds:02d}")
            self._elapsed_timer_id = self.safe_after(1000, self._tick_elapsed)
