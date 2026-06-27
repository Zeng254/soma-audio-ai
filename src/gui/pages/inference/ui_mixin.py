"""
Inference UI Mixin.

Contains all UI creation methods for the InferencePage.

Required attributes (initialized in InferencePage.__init__):
    - source_path: tk.StringVar - source audio file path
    - output_path: tk.StringVar - output file path
    - selected_model: tk.StringVar - selected model name
    - pitch_shift: tk.IntVar - pitch shift value
    - quality: tk.StringVar - quality preset
    - feature_extractor: tk.StringVar - feature extractor selection
    - f0_method: tk.StringVar - F0 extraction method
    - device: tk.StringVar - device selection
    - output_sample_rate: tk.StringVar - output sample rate
    - cluster_ratio: tk.DoubleVar - clustering ratio
    - separate_vocals: tk.BooleanVar - vocal separation toggle
    - dereverb_audio: tk.BooleanVar - dereverb toggle
    - separation_mode: tk.StringVar - separation mode for preprocessing
    - progress_var: tk.DoubleVar - progress bar value
    - status_var: tk.StringVar - status text
    - elapsed_var: tk.StringVar - elapsed time display
    - stage_var: tk.StringVar - current stage display
    - file_info_*: tk.StringVar - file info display variables
    - _last_directory: str - remembered directory for file dialogs
    - _settings: SettingsManager - settings manager instance
    - _model_cache: list - cached model list
    - _model_cache_time: float - cache timestamp

Methods provided by this mixin:
    - _create_widgets(parent)
    - _create_*_section(parent) - various UI sections
    - _browse_source(), _browse_output()
    - _on_source_path_changed()
    - _reset_file_info(), _load_file_info(path)
    - _update_fe_desc(), _update_f0_desc(), _update_dev_desc()
    - _update_cluster_label(value)
    - _refresh_models()
    - _find_model_file(model_name)
"""

import tkinter as tk
from tkinter import ttk, filedialog
import os
import threading
import time
from typing import Optional

from gui.styles import Colors, Fonts
from gui.utils import (
    SettingsManager, AUDIO_FILETYPES,
    FEATURE_EXTRACTORS, F0_METHODS, DEVICES, SAMPLE_RATES,
    DEFAULT_SAMPLE_RATE, DEFAULT_F0_METHOD, DEFAULT_FEATURE_EXTRACTOR,
    DEFAULT_DEVICE, PITCH_MIN, PITCH_MAX,
    MODEL_CACHE_TTL, MODEL_SEARCH_MAX_DEPTH,
)


class InferenceUIMixin:
    """Mixin class providing UI creation methods for InferencePage."""

    # Class constants (shared from constants module)
    FEATURE_EXTRACTORS = FEATURE_EXTRACTORS
    F0_METHODS = F0_METHODS
    DEVICES = DEVICES
    SAMPLE_RATES = SAMPLE_RATES

    # Quality presets (for backward compatibility)
    QUALITY_PRESETS = {
        "Standard": {"sample_rate": "40000", "description": "Good quality, faster"},
        "High": {"sample_rate": "44100", "description": "Better quality, moderate speed"},
        "Best": {"sample_rate": "48000", "description": "Highest quality, slower"},
    }

    def _create_widgets(self):
        """Create inference page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Song Cover Generation",
            "Convert audio using trained voice models"
        )

        # Main content area with two columns
        main_frame = ttk.Frame(self.content_frame, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left column - Configuration
        left_frame = ttk.Frame(main_frame, style="TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._create_dropzone_section(left_frame)
        self._create_file_info_section(left_frame)
        self._create_model_section(left_frame)
        self._create_basic_options_section(left_frame)
        self._create_advanced_options_section(left_frame)
        self._create_preprocessing_section(left_frame)
        self._create_output_section(left_frame)

        # Right column - Progress
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._create_progress_section(right_frame)
        self._create_log_section(right_frame)

    # ── Drop Zone ──────────────────────────────────────────────────────

    def _create_dropzone_section(self, parent: tk.Widget):
        """Create a visual drop zone for source audio files."""
        card = self.create_card(parent, "Source Audio")

        # Drop zone visual
        self.dropzone = tk.Frame(
            card, bg=Colors.BG_INPUT,
            highlightbackground=Colors.BORDER, highlightthickness=2,
            padx=20, pady=15,
        )
        self.dropzone.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            self.dropzone, text="\U0001f3a4",
            font=(Fonts.FAMILY, 28), bg=Colors.BG_INPUT, fg=Colors.TEXT_MUTED,
        ).pack(pady=(5, 0))

        tk.Label(
            self.dropzone,
            text="Click 'Browse' or drag dry vocal file here",
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
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

        ttk.Label(
            card, text="Supported: WAV, MP3, FLAC, OGG",
            style="Muted.TLabel"
        ).pack(anchor=tk.W, pady=(10, 0))

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
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{label_text}:", style="Card.TLabel", width=12).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=var, style="Muted.TLabel").pack(side=tk.LEFT, padx=(5, 0))

    # ── Model Selection ────────────────────────────────────────────────

    def _create_model_section(self, parent: tk.Widget):
        """Create model selection section."""
        card = self.create_card(parent, "Voice Model")

        model_frame = ttk.Frame(card, style="Card.TFrame")
        model_frame.pack(fill=tk.X)

        ttk.Label(model_frame, text="Model:", style="Card.TLabel").pack(side=tk.LEFT)

        self._model_combo = ttk.Combobox(
            model_frame, textvariable=self.selected_model, state="readonly"
        )
        self._model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        refresh_btn = ttk.Button(
            model_frame, text="\U0001f504",
            style="Secondary.TButton", command=self._refresh_models
        )
        refresh_btn.pack(side=tk.RIGHT, padx=(10, 0))

        self._refresh_models()

    # ── Basic Options ──────────────────────────────────────────────────

    def _create_basic_options_section(self, parent: tk.Widget):
        """Create basic conversion options (pitch, quality)."""
        card = self.create_card(parent, "Basic Options")

        # Pitch shift
        pitch_frame = ttk.Frame(card, style="Card.TFrame")
        pitch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pitch_frame, text="Pitch Shift:", style="Card.TLabel", width=14).pack(side=tk.LEFT)

        pitch_spinbox = tk.Spinbox(
            pitch_frame, from_=PITCH_MIN, to=PITCH_MAX,
            textvariable=self.pitch_shift,
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            buttonbackground=Colors.BG_TERTIARY, width=6
        )
        pitch_spinbox.pack(side=tk.LEFT)
        ttk.Label(pitch_frame, text="semitones (-12 to +12)", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Output sample rate
        sr_frame = ttk.Frame(card, style="Card.TFrame")
        sr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sr_frame, text="Sample Rate:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        sr_combo = ttk.Combobox(
            sr_frame, textvariable=self.output_sample_rate,
            values=self.SAMPLE_RATES, state="readonly", width=10
        )
        sr_combo.pack(side=tk.LEFT)
        ttk.Label(sr_frame, text="Hz (output)", style="Muted.TLabel").pack(side=tk.LEFT, padx=(10, 0))

    # ── Advanced Options ───────────────────────────────────────────────

    def _create_advanced_options_section(self, parent: tk.Widget):
        """Create advanced parameter controls."""
        card = self.create_card(parent, "Advanced Options")

        # Feature extractor
        fe_frame = ttk.Frame(card, style="Card.TFrame")
        fe_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fe_frame, text="Feature Extractor:", style="Card.TLabel", width=16).pack(side=tk.LEFT)
        fe_combo = ttk.Combobox(
            fe_frame, textvariable=self.feature_extractor,
            values=list(self.FEATURE_EXTRACTORS.keys()), state="readonly", width=14
        )
        fe_combo.pack(side=tk.LEFT)

        # Feature extractor description
        self.fe_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.fe_desc.pack(anchor=tk.W, pady=(0, 5))
        self._update_fe_desc()
        self.feature_extractor.trace_add("write", lambda *a: self._update_fe_desc())

        # F0 method
        f0_frame = ttk.Frame(card, style="Card.TFrame")
        f0_frame.pack(fill=tk.X, pady=5)
        ttk.Label(f0_frame, text="F0 Method:", style="Card.TLabel", width=16).pack(side=tk.LEFT)
        f0_combo = ttk.Combobox(
            f0_frame, textvariable=self.f0_method,
            values=list(self.F0_METHODS.keys()), state="readonly", width=14
        )
        f0_combo.pack(side=tk.LEFT)

        # F0 method description
        self.f0_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.f0_desc.pack(anchor=tk.W, pady=(0, 5))
        self._update_f0_desc()
        self.f0_method.trace_add("write", lambda *a: self._update_f0_desc())

        # Device
        dev_frame = ttk.Frame(card, style="Card.TFrame")
        dev_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dev_frame, text="Device:", style="Card.TLabel", width=16).pack(side=tk.LEFT)
        dev_combo = ttk.Combobox(
            dev_frame, textvariable=self.device,
            values=list(self.DEVICES.keys()), state="readonly", width=14
        )
        dev_combo.pack(side=tk.LEFT)

        # Device description
        self.dev_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.dev_desc.pack(anchor=tk.W, pady=(0, 5))
        self._update_dev_desc()
        self.device.trace_add("write", lambda *a: self._update_dev_desc())

        # Separator
        ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Clustering ratio slider
        cluster_frame = ttk.Frame(card, style="Card.TFrame")
        cluster_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cluster_frame, text="Cluster Ratio:", style="Card.TLabel", width=16).pack(side=tk.LEFT)

        self.cluster_scale = ttk.Scale(
            cluster_frame, from_=0.0, to=1.0,
            variable=self.cluster_ratio, orient=tk.HORIZONTAL,
        )
        self.cluster_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        self.cluster_value_label = ttk.Label(cluster_frame, text="0.0", style="Card.TLabel", width=5)
        self.cluster_value_label.pack(side=tk.LEFT)
        self.cluster_ratio.trace_add("write", self._update_cluster_label)

        ttk.Label(
            card,
            text="Timbre clustering ratio (0 = original, 1 = fully clustered)",
            style="Muted.TLabel"
        ).pack(anchor=tk.W, pady=(5, 0))

    # ── Preprocessing ──────────────────────────────────────────────────

    def _create_preprocessing_section(self, parent: tk.Widget):
        """Create audio preprocessing options."""
        card = self.create_card(parent, "Audio Preprocessing")

        # Separate vocals checkbox
        separate_frame = ttk.Frame(card, style="Card.TFrame")
        separate_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            separate_frame,
            text="Separate vocals first (recommended for songs with instruments)",
            variable=self.separate_vocals,
        ).pack(side=tk.LEFT)

        # Dereverb checkbox
        dereverb_frame = ttk.Frame(card, style="Card.TFrame")
        dereverb_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            dereverb_frame,
            text="Remove reverb (improves voice conversion quality)",
            variable=self.dereverb_audio,
        ).pack(side=tk.LEFT)

        # Separation mode (if separating)
        sep_mode_frame = ttk.Frame(card, style="Card.TFrame")
        sep_mode_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sep_mode_frame, text="Separation:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        sep_mode_combo = ttk.Combobox(
            sep_mode_frame, textvariable=self.separation_mode,
            values=["2-stem", "4-stem"], state="readonly", width=10
        )
        sep_mode_combo.pack(side=tk.LEFT)
        ttk.Label(sep_mode_frame, text="(vocals + accompaniment)", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

    # ── Output ─────────────────────────────────────────────────────────

    def _create_output_section(self, parent: tk.Widget):
        """Create output configuration section."""
        card = self.create_card(parent, "Output")

        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        ttk.Label(path_frame, text="Output:", style="Card.TLabel").pack(side=tk.LEFT)

        path_entry = ttk.Entry(path_frame, textvariable=self.output_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))

        browse_btn = ttk.Button(
            path_frame, text="Browse...",
            style="Secondary.TButton", command=self._browse_output
        )
        browse_btn.pack(side=tk.RIGHT)

        # Action buttons
        button_frame = ttk.Frame(card, style="Card.TFrame")
        button_frame.pack(fill=tk.X, pady=(15, 0))

        self.convert_btn = ttk.Button(
            button_frame, text="\u25b6 Start Conversion",
            style="Primary.TButton", command=self._start_conversion
        )
        self.convert_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(
            button_frame, text="\u23f9 Stop",
            style="Danger.TButton", command=self._stop_conversion,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)

    # ── Progress ───────────────────────────────────────────────────────

    def _create_progress_section(self, parent: tk.Widget):
        """Create multi-stage progress display."""
        card = self.create_card(parent, "Conversion Progress")

        # Stage indicator
        stage_frame = ttk.Frame(card, style="Card.TFrame")
        stage_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(stage_frame, text="Stage:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(stage_frame, textvariable=self.stage_var, style="Card.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            card, variable=self.progress_var,
            maximum=100, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Status row
        status_frame = ttk.Frame(card, style="Card.TFrame")
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, text="Status:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var, style="Card.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Elapsed time row
        time_frame = ttk.Frame(card, style="Card.TFrame")
        time_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(time_frame, text="Elapsed:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(time_frame, textvariable=self.elapsed_var, style="Card.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

    # ── Log ────────────────────────────────────────────────────────────

    def _create_log_section(self, parent: tk.Widget):
        """Create log output section."""
        card = self.create_card(parent, "Log Output")

        log_frame = ttk.Frame(card, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_display = tk.Text(
            log_frame, height=12,
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            wrap=tk.WORD, state=tk.DISABLED
        )
        self.log_display.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.log_display, command=self.log_display.yview)
        self.log_display.configure(yscrollcommand=scrollbar.set)

    # ── Browse Handlers ────────────────────────────────────────────────

    def _browse_source(self):
        """Open file dialog to select source audio."""
        filename = filedialog.askopenfilename(
            title="Select Source Audio",
            filetypes=AUDIO_FILETYPES,
            initialdir=self._last_directory,
        )

        if filename:
            self.source_path.set(filename)
            self._last_directory = os.path.dirname(filename)
            self._settings.set("inference_last_dir", self._last_directory)

            # Auto-generate output path
            base, ext = os.path.splitext(filename)
            self.output_path.set(f"{base}_cover.wav")

    def _browse_output(self):
        """Open file dialog to select output path."""
        filetypes = [
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("FLAC files", "*.flac"),
            ("All files", "*.*")
        ]

        filename = filedialog.asksaveasfilename(
            title="Save Output Audio",
            filetypes=filetypes,
            defaultextension=".wav",
            initialdir=self._last_directory,
        )

        if filename:
            self.output_path.set(filename)
            self._last_directory = os.path.dirname(filename)
            self._settings.set("inference_last_dir", self._last_directory)

    # ── File Info (background thread, fix #12) ─────────────────────────

    def _on_source_path_changed(self, *args):
        """Triggered when source_path variable changes."""
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
        """Load and display audio file information (runs in background thread)."""
        try:
            size_bytes = os.path.getsize(filepath)
            if size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / 1024 / 1024:.2f} MB"

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

    # ── Description Updaters ───────────────────────────────────────────

    def _update_fe_desc(self):
        fe = self.feature_extractor.get()
        self.fe_desc.configure(text=self.FEATURE_EXTRACTORS.get(fe, ""))

    def _update_f0_desc(self):
        f0 = self.f0_method.get()
        self.f0_desc.configure(text=self.F0_METHODS.get(f0, ""))

    def _update_dev_desc(self):
        dev = self.device.get()
        self.dev_desc.configure(text=self.DEVICES.get(dev, ""))

    def _update_cluster_label(self, *args):
        val = self.cluster_ratio.get()
        self.cluster_value_label.configure(text=f"{val:.2f}")

    # ── Model Management (with cache, fix #7, and depth limit, fix #10) ─

    def _refresh_models(self):
        """Refresh the list of available models with mtime-based cache."""
        now = time.time()
        if self._model_cache and (now - self._model_cache_time) < MODEL_CACHE_TTL:
            # Use cached results
            models = self._model_cache
        else:
            # Scan model directories
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

            # Cache results
            self._model_cache = models
            self._model_cache_time = now

        if not models:
            models = ["No models available"]

        self._available_models = models
        self._model_combo.configure(values=self._available_models)

        if self._available_models and self._available_models[0] != "No models available":
            self._model_combo.set(self._available_models[0])

    def _find_model_file(self, model_name: str) -> Optional[str]:
        """Find a model file by name in known model directories (depth-limited, fix #10)."""
        search_dirs = [
            os.path.join(os.path.expanduser("~"), ".soma", "models"),
            os.path.join(os.getcwd(), "assets", "models"),
        ]

        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue

            # Direct match first
            direct = os.path.join(search_dir, f"{model_name}.pth")
            if os.path.isfile(direct):
                return direct

            # Depth-limited recursive search (fix #10)
            for root, dirs, files in os.walk(search_dir):
                # Calculate current depth
                rel = os.path.relpath(root, search_dir)
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if depth >= MODEL_SEARCH_MAX_DEPTH:
                    dirs.clear()  # Prune deeper directories
                    continue

                for f in files:
                    if f == f"{model_name}.pth":
                        return os.path.join(root, f)

        return None
