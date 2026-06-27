"""
SeparationPage UI Mixin - widget creation and layout methods.

Contains all _create_* section methods, browse handlers, file info display,
and backend description updates.

Required attributes (initialized in SeparationPage.__init__):
    - source_path: tk.StringVar - source audio file path
    - output_dir: tk.StringVar - output directory path
    - separation_mode: tk.StringVar - separation mode selection
    - backend: tk.StringVar - backend selection
    - dereverb_enabled: tk.BooleanVar - dereverb toggle
    - output_format: tk.StringVar - output format selection
    - progress_var: tk.DoubleVar - progress bar value
    - status_var: tk.StringVar - status text
    - elapsed_var: tk.StringVar - elapsed time display
    - file_info_*: tk.StringVar - file info display variables
    - _last_directory: str - remembered directory for file dialogs
    - _settings: SettingsManager - settings manager instance

Methods provided by this mixin:
    - _create_widgets(parent)
    - _create_dropzone_section(parent)
    - _create_file_info_section(parent)
    - _create_mode_section(parent)
    - _create_output_section(parent)
    - _create_action_section(parent)
    - _create_progress_section(parent)
    - _create_log_section(parent)
    - _create_output_files_section(parent)
    - _browse_source()
    - _browse_output_dir()
    - _on_source_path_changed()
    - _reset_file_info()
    - _load_file_info(path)
    - _update_backend_desc()
"""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import os
from typing import Optional

from gui.styles import Colors, Fonts
from gui.utils import SettingsManager, AUDIO_FILETYPES


class SeparationUIMixin:
    """Mixin providing UI creation methods for SeparationPage."""

    # Class-level constants (accessed via self when mixed in)
    MODES: dict
    BACKENDS: dict
    OUTPUT_FORMATS: dict

    def _create_widgets(self):
        """Create separation page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Audio Separation",
            "Separate vocals, instruments, and other audio components"
        )

        # Main content area with two columns
        main_frame = ttk.Frame(self.content_frame, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left column - Configuration
        left_frame = ttk.Frame(main_frame, style="TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._create_dropzone_section(left_frame)
        self._create_file_info_section(left_frame)
        self._create_mode_section(left_frame)
        self._create_output_section(left_frame)
        self._create_action_section(left_frame)

        # Right column - Progress and Log
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._create_progress_section(right_frame)
        self._create_log_section(right_frame)
        self._create_output_files_section(right_frame)

    # ── Drop Zone ──────────────────────────────────────────────────────

    def _create_dropzone_section(self, parent: tk.Widget):
        """Create a visual drop zone for audio files."""
        card = self.create_card(parent, "源音频")

        # Drop zone frame with dashed border effect
        self.dropzone = tk.Frame(
            card,
            bg=Colors.BG_INPUT,
            highlightbackground=Colors.BORDER,
            highlightthickness=2,
            padx=20,
            pady=20,
        )
        self.dropzone.pack(fill=tk.X, pady=(0, 10))

        # Drop zone icon and text
        icon_label = tk.Label(
            self.dropzone,
            text="\U0001f4c1",  # folder emoji
            font=(Fonts.FAMILY, 28),
            bg=Colors.BG_INPUT,
            fg=Colors.TEXT_MUTED,
        )
        icon_label.pack(pady=(5, 0))

        hint_label = tk.Label(
            self.dropzone,
            text="点击[浏览]或拖入音频文件",
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            bg=Colors.BG_INPUT,
            fg=Colors.TEXT_SECONDARY,
        )
        hint_label.pack(pady=(0, 5))

        # Path entry with browse button
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        path_entry = ttk.Entry(path_frame, textvariable=self.source_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(
            path_frame, text="浏览...",
            style="Secondary.TButton",
            command=self._browse_source
        )
        browse_btn.pack(side=tk.RIGHT)

        # Help text
        ttk.Label(
            card,
            text="支持格式: WAV, MP3, FLAC, OGG",
            style="Muted.TLabel"
        ).pack(anchor=tk.W, pady=(10, 0))

        # Bind trace on source_path to update file info
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
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{label_text}:", style="Card.TLabel", width=12).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=var, style="Muted.TLabel").pack(side=tk.LEFT, padx=(5, 0))

    # ── Separation Settings ────────────────────────────────────────────

    def _create_mode_section(self, parent: tk.Widget):
        """Create separation mode and backend selection section."""
        card = self.create_card(parent, "分离设置")

        # Mode selection
        mode_frame = ttk.Frame(card, style="Card.TFrame")
        mode_frame.pack(fill=tk.X, pady=5)
        ttk.Label(mode_frame, text="模式:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(
            mode_frame, textvariable=self.separation_mode,
            values=list(self.MODES.keys()), state="readonly"
        )
        mode_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Backend selection
        backend_frame = ttk.Frame(card, style="Card.TFrame")
        backend_frame.pack(fill=tk.X, pady=5)
        ttk.Label(backend_frame, text="后端:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        backend_combo = ttk.Combobox(
            backend_frame, textvariable=self.backend,
            values=list(self.BACKENDS.keys()), state="readonly"
        )
        backend_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Backend description
        self.backend_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.backend_desc.pack(anchor=tk.W, pady=(5, 0))
        self._update_backend_desc()
        self.backend.trace_add("write", lambda *a: self._update_backend_desc())

        # Separator
        ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Dereverb toggle
        dereverb_frame = ttk.Frame(card, style="Card.TFrame")
        dereverb_frame.pack(fill=tk.X, pady=5)
        dereverb_check = ttk.Checkbutton(
            dereverb_frame,
            text="启用去混响 (减少房间回声)",
            variable=self.dereverb_enabled,
        )
        dereverb_check.pack(side=tk.LEFT)

        # Output format
        fmt_frame = ttk.Frame(card, style="Card.TFrame")
        fmt_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fmt_frame, text="输出格式:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        fmt_combo = ttk.Combobox(
            fmt_frame, textvariable=self.output_format,
            values=list(self.OUTPUT_FORMATS.keys()), state="readonly", width=10
        )
        fmt_combo.pack(side=tk.LEFT)

    # ── Output Directory ───────────────────────────────────────────────

    def _create_output_section(self, parent: tk.Widget):
        """Create output directory selection section."""
        card = self.create_card(parent, "输出目录")

        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        path_entry = ttk.Entry(path_frame, textvariable=self.output_dir)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(
            path_frame, text="浏览...",
            style="Secondary.TButton",
            command=self._browse_output_dir
        )
        browse_btn.pack(side=tk.RIGHT)

        ttk.Label(
            card,
            text="输出文件自动命名: <源文件名>_<类型>.<格式>",
            style="Muted.TLabel"
        ).pack(anchor=tk.W, pady=(10, 0))

    # ── Action Buttons ─────────────────────────────────────────────────

    def _create_action_section(self, parent: tk.Widget):
        """Create action buttons section."""
        button_frame = ttk.Frame(parent, style="TFrame")
        button_frame.pack(fill=tk.X, pady=20)

        self.start_btn = ttk.Button(
            button_frame, text="\u25b6 Start Separation",
            style="Primary.TButton",
            command=self._start_separation
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(
            button_frame, text="\u23f9 Stop",
            style="Danger.TButton",
            command=self._stop_separation,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)

    # ── Progress ───────────────────────────────────────────────────────

    def _create_progress_section(self, parent: tk.Widget):
        """Create progress display section with elapsed time."""
        card = self.create_card(parent, "进度")

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            card, variable=self.progress_var,
            maximum=100, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Status row
        status_frame = ttk.Frame(card, style="Card.TFrame")
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, text="状态:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var, style="Card.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Elapsed time row
        time_frame = ttk.Frame(card, style="Card.TFrame")
        time_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(time_frame, text="耗时:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(time_frame, textvariable=self.elapsed_var, style="Card.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

    # ── Log ────────────────────────────────────────────────────────────

    def _create_log_section(self, parent: tk.Widget):
        """Create log output section."""
        card = self.create_card(parent, "日志输出")

        log_frame = ttk.Frame(card, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_display = tk.Text(
            log_frame, height=10,
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            wrap=tk.WORD, state=tk.DISABLED
        )
        self.log_display.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.log_display, command=self.log_display.yview)
        self.log_display.configure(yscrollcommand=scrollbar.set)

    # ── Output Files ───────────────────────────────────────────────────

    def _create_output_files_section(self, parent: tk.Widget):
        """Create output files list section."""
        card = self.create_card(parent, "输出文件")

        self.output_listbox = tk.Listbox(
            card, height=5,
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            selectbackground=Colors.ACCENT_PRIMARY,
            selectforeground=Colors.BG_PRIMARY,
        )
        self.output_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        open_btn = ttk.Button(
            card, text="\U0001f4c1 Open Output Folder",
            style="Secondary.TButton",
            command=self._open_output_folder
        )
        open_btn.pack(anchor=tk.W)

    # ── Browse Handlers ────────────────────────────────────────────────

    def _browse_source(self):
        """Browse for source audio file, remembering last directory."""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=AUDIO_FILETYPES,
            initialdir=self._last_directory,
        )

        if filename:
            self.source_path.set(filename)
            # Remember directory via SettingsManager
            self._last_directory = os.path.dirname(filename)
            self._settings.set("separation_last_dir", self._last_directory)

            # Auto-set output directory if empty
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(filename))

    def _browse_output_dir(self):
        """Browse for output directory, remembering last directory."""
        dirname = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self._last_directory,
        )
        if dirname:
            self.output_dir.set(dirname)
            self._last_directory = dirname
            self._settings.set("separation_last_dir", dirname)

    # ── File Info Update (background thread) ───────────────────────────

    def _on_source_path_changed(self, *args):
        """Triggered when source_path variable changes."""
        filepath = self.source_path.get()
        if not filepath:
            self._reset_file_info()
            return
        if not os.path.isfile(filepath):
            self._reset_file_info()
            return
        # Update file info in background thread (fix #12)
        threading.Thread(target=self._load_file_info, args=(filepath,), daemon=True).start()

    def _reset_file_info(self):
        """Reset file info to defaults."""
        self.file_info_filename.set("未选择文件")
        self.file_info_duration.set("--")
        self.file_info_samplerate.set("--")
        self.file_info_channels.set("--")
        self.file_info_filesize.set("--")

    def _load_file_info(self, filepath: str):
        """Load and display audio file information (runs in background thread)."""
        try:
            # File size
            size_bytes = os.path.getsize(filepath)
            if size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / 1024 / 1024:.2f} MB"

            filename = os.path.basename(filepath)
            self.safe_after(0, lambda: self.file_info_filename.set(filename))
            self.safe_after(0, lambda: self.file_info_filesize.set(size_str))

            # Try to read audio metadata
            try:
                import soundfile as sf
                info = sf.info(filepath)
                duration_sec = info.duration
                minutes = int(duration_sec // 60)
                seconds = int(duration_sec % 60)
                duration_str = f"{minutes}:{seconds:02d}"

                self.safe_after(0, lambda: self.file_info_duration.set(duration_str))
                self.safe_after(0, lambda: self.file_info_samplerate.set(f"{info.samplerate} Hz"))
                ch_map = {1: "单声道", 2: "立体声"}
                self.safe_after(0, lambda: self.file_info_channels.set(
                    ch_map.get(info.channels, f"{info.channels} ch")
                ))
            except Exception:
                self.safe_after(0, lambda: self.file_info_duration.set("N/A"))
                self.safe_after(0, lambda: self.file_info_samplerate.set("N/A"))
                self.safe_after(0, lambda: self.file_info_channels.set("N/A"))

        except Exception:
            self.safe_after(0, self._reset_file_info)

    # ── Backend Description ────────────────────────────────────────────

    def _update_backend_desc(self):
        """Update backend description label."""
        backend = self.backend.get()
        desc = self.BACKENDS.get(backend, "")
        self.backend_desc.configure(text=desc)
