"""
Comparison page - UI Mixin.

Contains all UI creation methods for the comparison page.

Required attributes (initialized in ComparisonPage.__init__):
    - source_path: tk.StringVar - source audio file path
    - output_dir: tk.StringVar - output directory path
    - selected_model: tk.StringVar - selected model name
    - pitch_shift: tk.IntVar - pitch shift value
    - feature_extractor: tk.StringVar - feature extractor selection
    - f0_method: tk.StringVar - F0 extraction method
    - device: tk.StringVar - device selection
    - output_sample_rate: tk.StringVar - output sample rate
    - cluster_ratio: tk.DoubleVar - clustering ratio
    - elapsed_var: tk.StringVar - elapsed time display
    - file_info_*: tk.StringVar - file info display variables
    - _last_directory: str - remembered directory for file dialogs
    - _settings: SettingsManager - settings manager instance
    - _model_cache: list - cached model list
    - _model_cache_time: float - cache timestamp

Methods provided by this mixin:
    - _create_widgets(parent)
    - _create_*_section(parent) - various UI sections
    - _browse_source(), _browse_output_dir()
    - _on_source_path_changed()
    - _reset_file_info(), _load_file_info(path)
    - _refresh_models()
    - _find_comboboxes(widget)
    - _find_model_file(model_name)
    - _update_cluster_label(value)
"""

import tkinter as tk
from tkinter import ttk, filedialog
import os
import time
import threading
from typing import List, Optional

from gui.styles import Colors, Fonts
from gui.utils import (
    AUDIO_FILETYPES, FEATURE_EXTRACTORS, F0_METHODS, DEVICES, SAMPLE_RATES,
    DEFAULT_SAMPLE_RATE, DEFAULT_F0_METHOD, DEFAULT_FEATURE_EXTRACTOR,
    DEFAULT_DEVICE, PITCH_MIN, PITCH_MAX,
    MODEL_CACHE_TTL, MODEL_SEARCH_MAX_DEPTH,
)

# Column definitions for task list
COLUMNS = ("id", "model", "pitch", "f0", "feature", "status", "time")
COLUMN_HEADINGS = {
    "id": "序号", "model": "模型", "pitch": "音调",
    "f0": "F0", "feature": "特征", "status": "状态", "time": "耗时",
}
COLUMN_WIDTHS = {
    "id": 40, "model": 120, "pitch": 50,
    "f0": 60, "feature": 70, "status": 90, "time": 60,
}


class ComparisonUIMixin:
    """Mixin class for comparison page UI creation methods."""

    # Use shared constants
    FEATURE_EXTRACTORS = FEATURE_EXTRACTORS
    F0_METHODS = F0_METHODS
    DEVICES = DEVICES
    SAMPLE_RATES = SAMPLE_RATES

    def _create_widgets(self):
        """Create comparison page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "效果对比",
            "使用不同参数对比声音转换结果"
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
        card = self.create_card(parent, "源音频")

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
            text="点击[浏览]或拖入音频文件",
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            bg=Colors.BG_INPUT, fg=Colors.TEXT_SECONDARY,
        ).pack(pady=(0, 5))

        # Path entry with browse button
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        path_entry = ttk.Entry(path_frame, textvariable=self.source_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(
            path_frame, text="浏览...",
            style="Secondary.TButton", command=self._browse_source
        )
        browse_btn.pack(side=tk.RIGHT)

        # Bind trace to update file info
        self.source_path.trace_add("write", self._on_source_path_changed)

    # ── File Info ──────────────────────────────────────────────────────

    def _create_file_info_section(self, parent: tk.Widget):
        """Create file information display section."""
        card = self.create_card(parent, "文件信息")

        info_frame = ttk.Frame(card, style="Card.TFrame")
        info_frame.pack(fill=tk.X)

        info_items = [
            ("文件名", self.file_info_filename),
            ("时长", self.file_info_duration),
            ("采样率", self.file_info_samplerate),
            ("声道", self.file_info_channels),
            ("文件大小", self.file_info_filesize),
        ]

        for label_text, var in info_items:
            row = ttk.Frame(info_frame, style="Card.TFrame")
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label_text}:", style="Card.TLabel", width=12).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=var, style="Muted.TLabel").pack(side=tk.LEFT, padx=(5, 0))

    # ── Task Configuration ─────────────────────────────────────────────

    def _create_task_config_section(self, parent: tk.Widget):
        """Create task parameter configuration section."""
        card = self.create_card(parent, "任务配置")

        # Model selection
        model_frame = ttk.Frame(card, style="Card.TFrame")
        model_frame.pack(fill=tk.X, pady=3)
        ttk.Label(model_frame, text="模型:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
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
        ttk.Label(pitch_frame, text="音调:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        pitch_spinbox = tk.Spinbox(
            pitch_frame, from_=PITCH_MIN, to=PITCH_MAX,
            textvariable=self.pitch_shift,
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            buttonbackground=Colors.BG_TERTIARY, width=5
        )
        pitch_spinbox.pack(side=tk.LEFT)
        ttk.Label(pitch_frame, text="半音", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Feature extractor
        fe_frame = ttk.Frame(card, style="Card.TFrame")
        fe_frame.pack(fill=tk.X, pady=3)
        ttk.Label(fe_frame, text="特征:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        fe_combo = ttk.Combobox(
            fe_frame, textvariable=self.feature_extractor,
            values=list(self.FEATURE_EXTRACTORS.keys()), state="readonly", width=12
        )
        fe_combo.pack(side=tk.LEFT)

        # F0 method
        f0_frame = ttk.Frame(card, style="Card.TFrame")
        f0_frame.pack(fill=tk.X, pady=3)
        ttk.Label(f0_frame, text="F0方法:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        f0_combo = ttk.Combobox(
            f0_frame, textvariable=self.f0_method,
            values=list(self.F0_METHODS.keys()), state="readonly", width=12
        )
        f0_combo.pack(side=tk.LEFT)

        # Device
        dev_frame = ttk.Frame(card, style="Card.TFrame")
        dev_frame.pack(fill=tk.X, pady=3)
        ttk.Label(dev_frame, text="设备:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        dev_combo = ttk.Combobox(
            dev_frame, textvariable=self.device,
            values=list(self.DEVICES.keys()), state="readonly", width=12
        )
        dev_combo.pack(side=tk.LEFT)

        # Sample rate
        sr_frame = ttk.Frame(card, style="Card.TFrame")
        sr_frame.pack(fill=tk.X, pady=3)
        ttk.Label(sr_frame, text="采样率:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        sr_combo = ttk.Combobox(
            sr_frame, textvariable=self.output_sample_rate,
            values=self.SAMPLE_RATES, state="readonly", width=12
        )
        sr_combo.pack(side=tk.LEFT)

        # Cluster ratio
        cluster_frame = ttk.Frame(card, style="Card.TFrame")
        cluster_frame.pack(fill=tk.X, pady=3)
        ttk.Label(cluster_frame, text="聚类:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
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
            clear_frame, text="清除已完成",
            style="Secondary.TButton", command=self._clear_done_tasks
        )
        clear_done_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        remove_btn = ttk.Button(
            clear_frame, text="删除选中",
            style="Secondary.TButton", command=self._remove_selected_task
        )
        remove_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        # Config save/load
        config_frame = ttk.Frame(card, style="Card.TFrame")
        config_frame.pack(fill=tk.X, pady=(10, 0))

        save_cfg_btn = ttk.Button(
            config_frame, text="保存配置",
            style="Secondary.TButton", command=self._save_config
        )
        save_cfg_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        load_cfg_btn = ttk.Button(
            config_frame, text="加载配置",
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
        card = self.create_card(parent, "结果日志")

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
        card = self.create_card(parent, "导出结果")

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
        self.file_info_filename.set("未选择文件")
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
